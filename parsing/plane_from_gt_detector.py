import torch
from torch import nn
from parsing.backbones import build_backbone
from parsing.encoder.plane_encoder import PlaneEncoder
# from epnet.structures.linelist_ops import linesegment_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from parsing.utils.labels import LabelMapper
from parsing.modules.gnn import WireframeGNNHead
from parsing.modules.line_detection import LineClassifier
from parsing.modules.plane_detection import PlaneClassifier
from parsing.modules.plane_detection_gnn import PlaneClassifierGNN
from parsing.detector import get_junctions, non_maximum_suppression
from parsing.utils.loss import sigmoid_l1_loss
import random
from parsing.utils.logger import CudaTimer




class PlaneFromGtDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PlaneEncoder(cfg)
        self.backbone = build_backbone(cfg)
        self.output_idx = np.cumsum([0] + [h[0] for h in cfg.MODEL.HEAD_SIZE])
        if getattr(cfg.MODEL.PLANE_HEAD, 'CENTROID_LOSS_WEIGHTS', None):
            pc_label_weights = torch.tensor(cfg.MODEL.PLANE_HEAD.CENTROID_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            pc_label_weights = None
        self.pc_label_loss = nn.CrossEntropyLoss(weight = pc_label_weights)

        if cfg.MODEL.PLANE_HEAD.STRATEGY == "graph":
            self.plane_classifier = PlaneClassifierGNN(cfg)
        else:
            self.plane_classifier = PlaneClassifier.create(cfg)

        self.train_step = 0
        self.enable_timing = cfg.MODEL.ENABLE_TIMING
        self.n_out_plane_centroid = 30
        self.nbr_line_labels = len(cfg.MODEL.LINE_LABELS)
        self.negative_line_ratio = cfg.MODEL.NEGATIVE_LINE_RATIO

        # DEBUG
        # from parsing.utils.visualization import ImagePlotter
        # lm = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        # self.img_viz = ImagePlotter(lm.get_line_labels(), lm.get_junction_labels(), cfg.MODEL.PLANE_LABELS)


    def forward(self, images, annotations = None, output_features = False):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations, output_features=output_features)

    def _get_gt_junctions_and_lines(self, ann, device):
        output = {}
        # extra_info = {}
        juncs_pred = ann['junctions'].to(device)
        juncs_pred[:,0] *= 128/float(ann['width'])
        juncs_pred[:,1] *= 128/float(ann['height'])
        juncs_label = ann['junctions_semantic']
        # juncs_score = torch.zeros([juncs_pred.size(0), len(cfg.JUNCTION_LABELS)])
        # juncs_score[range(juncs_label.size(0)), juncs_label] = 1
        # juncs_logits = juncs_score
        output.update({
            'juncs_pred': juncs_pred,
            'juncs_label': juncs_label,
            'juncs_valid_score': torch.ones(juncs_pred.size(0)),
            'juncs_label_score': torch.ones(juncs_pred.size(0)),
        #     'juncs_score': juncs_score,
        })
        # extra_info.update({
        #     'junc_prior_ver': juncs_pred
        # })

        # junctions = ann['junctions']
        # junctions[:,0] *= 128/float(ann['width'])
        # junctions[:,1] *= 128/float(ann['height'])
        edges_positive = ann['edges_positive']
        edges_negative = ann['edges_negative']
        lines_label = ann['edges_semantic']
        nbr_negative_edges = min(int(self.negative_line_ratio*len(edges_positive)),edges_negative.size(0))
        if nbr_negative_edges > 0:
            rand_idx = torch.randperm(edges_negative.size(0), device=edges_negative.device)[:nbr_negative_edges]
            edges = torch.cat((edges_positive, edges_negative[rand_idx]), dim=0)
            lines_label = torch.cat((lines_label, torch.zeros(nbr_negative_edges, dtype = lines_label.dtype, device = lines_label.device)))
        else:
            edges = edges_positive
        lines_score = torch.zeros([edges.size(0), self.nbr_line_labels], device=device)
        valid_mask = lines_label>0
        lines_score[valid_mask,lines_label[valid_mask]] = 1
        lines_score[~valid_mask,0] = 0.5
        lines_pred = torch.cat((juncs_pred[edges[:,0]], juncs_pred[edges[:,1]]),dim=-1).to(device)
        lines_valid_score = 1 - lines_score[:,0]
        lines_label_score = torch.gather(lines_score, 1, lines_label.unsqueeze(1)).squeeze(1)
        lines_label = torch.ones_like(lines_label)
        # extra_info['lines_prior_ver'] = extra_info['lines_prior_scoring'] = lines_pred
        output.update({
            'lines_pred': lines_pred,
            'lines_label': lines_label,
            'lines_valid_score': lines_valid_score,
            'lines_label_score': lines_label_score,
            # 'lines_score': lines_score,
            'lines_logits': lines_score,
            'edges_pred': edges,
            'num_proposals': 0,
        })
        # extra_info.update({
        #     'lines_prior_scoring': lines_pred,
        #     'lines_prior_ver': lines_pred
        # })
        return output#, extra_info


    def _get_output_dist(self, output):
        pc_label_out= output[:,self.output_idx[0]:self.output_idx[1]]
        pc_off_out= output[:,self.output_idx[1]:self.output_idx[2]]
        return pc_label_out, pc_off_out


    def forward_test(self, images, annotations = None, output_features = False):
        device = images.device

        extra_info = {
            'time_backbone': 0.0,
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }

        extra_info['time_backbone'] = CudaTimer.initiate_timer(active=self.enable_timing)
        outputs, features = self.backbone(images)
        extra_info['time_backbone'] = extra_info['time_backbone'].end_timer()

        pc_label_out, pc_off_out = self._get_output_dist(outputs[0])
        pc_label_prob = pc_label_out.softmax(1)
        pc_loc_pred = 1-pc_label_prob[:,0,None]
        pc_off_pred = pc_off_out.sigmoid() - 0.5

        pc_loc_pred_nms = non_maximum_suppression(pc_loc_pred[0])
        topK = min(self.n_out_plane_centroid, int((pc_loc_pred_nms>0.008).float().sum().item()))

        # Reuse get_junctions since we setup the plane centroid the same way
        pcentroid_pred, pcentroid_valid_score, flat_index = get_junctions(pc_loc_pred_nms, pc_off_pred[0], topk=topK)
        pcentroid_logits = (pc_label_out.flatten(start_dim=2)[0,:,flat_index]).T
        pcentroid_score = (pc_label_prob.flatten(start_dim=2)[0,:,flat_index]).T
        pcentroid_label = pcentroid_score.argmax(dim=1)

        pcentroid_valid_score = 1-pcentroid_score[:,0]
        pcentroid_label_score = torch.gather(pcentroid_score, 1, pcentroid_label.unsqueeze(1)).squeeze(1)


        batch_size = pc_label_out.size(0)
        assert batch_size == 1
        ann = annotations[0]

        jl_output = self._get_gt_junctions_and_lines(ann, device)

        some_lines_valid = jl_output['lines_pred'].numel() > 0
        if some_lines_valid:
            juncs_pred = jl_output['juncs_pred']
            # Plane classification
            extra_info['time_plane_classifier_total'] = CudaTimer.initiate_timer(active=self.enable_timing)
            output_planes, extra_info_planes = self.plane_classifier.forward_test(juncs_pred, jl_output, features, img=images[0])
            extra_info['time_plane_classifier_total'] = extra_info['time_plane_classifier_total'].end_timer()


        if annotations:
            width = annotations[0]['width']
            height = annotations[0]['height']
        else:
            width = images.size(3)
            height = images.size(2)

        sx = width/features.size(3)
        sy = height/features.size(2)

        output = {
            'num_proposals': 0,
            'filename': annotations[0]['filename'] if annotations else None,
            'width': width,
            'height': height,
        }

        output['pcentroid_pred'] = pcentroid_pred
        output['pcentroid_label_score'] = pcentroid_label_score
        output['pcentroid_valid_score'] = pcentroid_valid_score
        output['pcentroid_score'] = pcentroid_score
        output['pcentroid_label'] = pcentroid_label

        if some_lines_valid:
            output.update(jl_output)
            output.update(output_planes)
            output['juncs_pred'][:,0] *= sx
            output['juncs_pred'][:,1] *= sy
            # output['junc_prior_ver'] = output['juncs_pred']

            output['lines_pred'][:,0] *= sx
            output['lines_pred'][:,1] *= sy
            output['lines_pred'][:,2] *= sx
            output['lines_pred'][:,3] *= sy

            output['pcentroid_pred'][:,0] *= sx
            output['pcentroid_pred'][:,1] *= sy
            # extra_info['lines_prior_ver'] = extra_info['lines_prior_scoring'] = output['lines_pred']



            output.update(output_planes)
            extra_info.update(extra_info_planes)
        else:
            output.update({
                'lines_pred': torch.tensor([]),
                'juncs_pred': torch.tensor([])
                })

        return output, extra_info

    def forward_train(self, images, annotations = None):
        device = images.device

        extra_info = {
            'time_encoder': 0.0,
            'time_backbone':  0.0,
            'time_batch_loss':  0.0,
            'time_line_classifier_total':  0.0,
            'time_plane_classifier_total':  0.0,
            }

        self.train_step += 1

        extra_info['time_encoder'] = CudaTimer.initiate_timer(active=self.enable_timing)
        targets = self.encoder(annotations)
        extra_info['time_encoder'] = extra_info['time_encoder'].end_timer()

        extra_info['time_backbone'] = CudaTimer.initiate_timer(active=self.enable_timing)
        outputs, features = self.backbone(images)
        extra_info['time_backbone'] = extra_info['time_backbone'].end_timer()

        loss_dict = {
            'loss_pc_label': torch.zeros(1, device=device),
            'loss_pc_off': torch.zeros(1, device=device),
        }
        loss_dict.update(self.plane_classifier.initialize_loss(device))

        # DEBUG encoder
        # from parsing.utils.comm import to_device
        # for i, ann in enumerate(annotations[:2]):
        #     ann_cpu = to_device(ann, 'cpu')
        #     print('images',images.shape)
        #     print('targets[pc_mask]',targets['pc_mask'].shape)
        #     self.img_viz.plot_gt_image(images[i].to('cpu'), ann_cpu, '/host_home/debug/plane_centroid', desc = f'GT_step{self.train_step}_image{i}')
        #     del ann_cpu['planes']
        #     # self.img_viz.no_border_imshow(img,dpi=dpi)
        #     self.img_viz.plot_gt_image(targets['pc_mask'][i].to('cpu').to(int), ann_cpu, '/host_home/debug/plane_centroid', desc = f'GT_mask_step{self.train_step}_image{i}', show_legend=False)
        #----------------

        extra_info['time_batch_loss'] = CudaTimer.initiate_timer(active=self.enable_timing)
        for nstack, output in enumerate(outputs):
            pc_label_out, pc_off_out = self._get_output_dist(output)
            loss_dict['loss_pc_label'] += self.pc_label_loss(
                pc_label_out.flatten(start_dim=2),
                targets['pc_label'].flatten(start_dim=1),
                )
            loss_dict['loss_pc_off'] += sigmoid_l1_loss(pc_off_out, targets['pc_off'], -0.5, targets['pc_mask'])

        pc_label_out, pc_off_out = self._get_output_dist(outputs[0])
        pc_label_prob = pc_label_out.softmax(1)
        pc_label = pc_label_prob.argmax(1)
        pc_loc_pred = 1-pc_label_prob[:,0,None]
        pc_off_pred= pc_off_out.sigmoid() - 0.5

        # ------------------- DEBGU ------------------------
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.title('Logits')
        # plt.plot(pc_label_out[0].to('cpu').detach().flatten(start_dim=1).numpy().T)
        # plt.legend(range(4))
        #
        # plt.subplot(2,2,2)
        # plt.title('Score')
        # plt.plot(pc_label_prob[0].to('cpu').detach().flatten(start_dim=1).numpy().T)
        #
        # plt.subplot(2,2,3)
        # plt.title('Label')
        # plt.plot(pc_label[0].to('cpu').detach().flatten().numpy())
        # plt.plot(targets['pc_label'][0].to('cpu').detach().flatten().numpy(), '--')
        # plt.legend(['Pred', 'GT'])

        # plt.savefig(f'/host_home/debug/centroid_loss/logits_original_train.pdf')
        # ------------------- DEBUG END ------------------------

        # Plane batch loss
        self.plane_classifier.forward_train_batch(features)

        extra_info['time_batch_loss'] = extra_info['time_batch_loss'].end_timer()

        batch_size = len(annotations)
        meta = {}
        for i, ann in enumerate(annotations):
            jl_output = self._get_gt_junctions_and_lines(ann, device)
            juncs_pred = jl_output['juncs_pred']

            if jl_output['lines_pred'].size(0) > 0:
                t = CudaTimer.initiate_timer(active=self.enable_timing)
                self.plane_classifier.forward_train(i, juncs_pred, jl_output, meta, ann, batch_size, img=images[i], batch_idx = self.train_step)
                extra_info['time_plane_classifier_total'] += t.end_timer()


        # Update with plane detection losses
        plane_loss_dict, plane_extra_info = self.plane_classifier.reset_and_get_loss_dict()
        loss_dict.update(plane_loss_dict)
        extra_info.update(plane_extra_info)

        return loss_dict, extra_info
