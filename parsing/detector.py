import torch
from torch import nn
from parsing.backbones import build_backbone
from parsing.encoder.hafm import HAFMencoder
# from epnet.structures.linelist_ops import linesegment_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from parsing.utils.labels import LabelMapper
from parsing.utils.loss import sigmoid_l1_loss
from parsing.modules.gnn import WireframeGNNHead
from parsing.modules.line_detection import LineClassifier
from parsing.modules.line_generation import LineGenerator
from parsing.modules.plane_detection import PlaneClassifier
from parsing.modules.plane_detection_gnn import PlaneClassifierGNN
import random
from parsing.utils.logger import CudaTimer


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc_flat = jloc.flatten()
    joff_flat = joff.flatten(start_dim=1)

    scores, index = torch.topk(jloc_flat, k=topk)
    y = (index / width).float() + torch.gather(joff_flat[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff_flat[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    score_mask = scores>th

    return junctions[score_mask], scores[score_mask], index[score_mask]

class WireframeDetector(nn.Module):
    def __init__(self, cfg):
        super(WireframeDetector, self).__init__()
        self.hafm_encoder = HAFMencoder(cfg)
        self.backbone = build_backbone(cfg)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.use_gt_junctions = cfg.MODEL.USE_GT_JUNCTIONS
        self.output_idx = np.cumsum([0] + [h[0] for h in cfg.MODEL.HEAD_SIZE])

        if cfg.MODEL.PARSING_HEAD.STRATEGY == "line_generator":
            self.line_classifier = LineGenerator(cfg)
            self._get_output_dist = self._get_output_dist_line_generator
        elif cfg.MODEL.PARSING_HEAD.STRATEGY == "line_classifier":
            self.line_classifier = LineClassifier(cfg)
            self._get_output_dist = self._get_output_dist_line_classifier
        else:
            raise NotImplementedError(f'Line sampling strategy {cfg.MODEL.PARSING_HEAD.STRATEGY} not implemented')

        if cfg.MODEL.PLANE_HEAD.STRATEGY == "graph":
            self.plane_classifier = PlaneClassifierGNN(cfg)
        else:
            self.plane_classifier = PlaneClassifier.create(cfg)

        # self.gnn_head = WireframeGNNHead(cfg)

        if getattr(cfg.MODEL, 'JUNCTION_LOSS_WEIGHTS', None):
            junction_label_weights = torch.tensor(cfg.MODEL.JUNCTION_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            junction_label_weights = None
        self.junction_label_loss = nn.CrossEntropyLoss(weight = junction_label_weights)
        # self.gnn_junction_label_loss = nn.CrossEntropyLoss()


        self.train_step = 0
        self.enable_timing = cfg.MODEL.ENABLE_TIMING


    def _get_output_dist_line_classifier(self, output):
        jlabel_out= output[:,self.output_idx[3]:self.output_idx[4]]
        joff_out= output[:,self.output_idx[4]:self.output_idx[5]]
        return jlabel_out, joff_out

    def _get_output_dist_line_generator(self, output):
        jlabel_out= output[:,self.output_idx[0]:self.output_idx[1]]
        joff_out= output[:,self.output_idx[1]:self.output_idx[2]]
        return jlabel_out, joff_out

    def forward(self, images, annotations = None, output_features = False):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images, annotations=annotations, output_features=output_features)

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

        jlabel_out, joff_out = self._get_output_dist(outputs[0])
        jlabel_prob = jlabel_out.softmax(1)
        jloc_pred = 1-jlabel_prob[:,0,None]
        joff_pred= joff_out.sigmoid() - 0.5


        batch_size = jlabel_out.size(0)
        assert batch_size == 1
        ann = annotations[0]

        extra_info['time_proposal'] = CudaTimer.initiate_timer(active=self.enable_timing)

        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
        topK = min(self.n_out_junc, int((jloc_pred_nms>0.008).float().sum().item()))

        if self.use_gt_junctions:
            juncs_pred = ann['junctions'].to(device)
            # juncs_pred[:,0] *= 128/float(ann['width'])
            # juncs_pred[:,1] *= 128/float(ann['height'])
            juncs_label = ann['junctions_semantic']
            juncs_score = torch.zeros([juncs_pred.size(0), jlabel_prob.size(1)])
            juncs_score[range(juncs_label.size(0)), juncs_label] = 1
            juncs_logits = juncs_score
        else:
            juncs_pred, juncs_valid_score, flat_index = get_junctions(jloc_pred_nms, joff_pred[0], topk=topK)
            juncs_logits = (jlabel_out.flatten(start_dim=2)[0,:,flat_index]).T
            juncs_score = (jlabel_prob.flatten(start_dim=2)[0,:,flat_index]).T
            juncs_label = juncs_score.argmax(dim=1)
            # junction_features = loi_features[0].flatten(start_dim=1)[:,flat_index].T


        extra_info['time_proposal'] = extra_info['time_proposal'].end_timer()
        extra_info['time_line_classifier_total'] = CudaTimer.initiate_timer(active=self.enable_timing)
        output_lines, extra_info_lines, line_feature_space = self.line_classifier.forward_test(outputs, features, juncs_pred, annotations)
        extra_info['time_line_classifier_total'] = extra_info['time_line_classifier_total'].end_timer()

        some_lines_valid = output_lines['lines_pred'].numel() > 0
        if some_lines_valid:
            valid_junction_idx = extra_info_lines['valid_junction_idx']
            del extra_info_lines['valid_junction_idx']

            juncs_final = juncs_pred[valid_junction_idx]
            # junction_features = junction_features[valid_junction_idx]
            juncs_logits = juncs_logits[valid_junction_idx]

            juncs_score = juncs_logits.softmax(1)
            juncs_label = juncs_score.argmax(1)
            juncs_valid_score = 1-juncs_score[:,0]
            juncs_label_score = torch.gather(juncs_score, 1, juncs_label.unsqueeze(1)).squeeze(1)


            # Plane classification
            extra_info['time_plane_classifier_total'] = CudaTimer.initiate_timer(active=self.enable_timing)
            output_planes, extra_info_planes = self.plane_classifier.forward_test(juncs_final, output_lines, features, img=images[0])
            extra_info['time_plane_classifier_total'] = extra_info['time_plane_classifier_total'].end_timer()


        if annotations:
            width = annotations[0]['width']
            height = annotations[0]['height']
        else:
            width = images.size(3)
            height = images.size(2)

        sx = width/jloc_pred.size(3)
        sy = height/jloc_pred.size(2)

        juncs_pred[:,0] *= sx
        juncs_pred[:,1] *= sy
        extra_info['junc_prior_ver'] = juncs_pred

        output = {
            'num_proposals': 0,
            'filename': annotations[0]['filename'] if annotations else None,
            'width': width,
            'height': height,
        }
        output.update(output_lines)
        extra_info.update(extra_info_lines)


        if some_lines_valid:
            juncs_final[:,0] *= sx
            juncs_final[:,1] *= sy

            output.update({
                'juncs_pred': juncs_final,
                'juncs_label': juncs_label,
                'juncs_valid_score': juncs_valid_score,
                'juncs_label_score': juncs_label_score,
                'juncs_score': juncs_score,
            })
            output.update(output_planes)
            extra_info.update(extra_info_planes)
        else:
            output.update({
                'lines_pred': torch.tensor([]),
                'juncs_pred': torch.tensor([])
                })

        if 'pcentroid_pred' in output:
            output['pcentroid_pred'][:,0] *= sx
            output['pcentroid_pred'][:,1] *= sy

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

        # TODO: Caching the encoding and implement transforms for it might speed up training
        extra_info['time_encoder'] = CudaTimer.initiate_timer(active=self.enable_timing)
        targets , metas = self.hafm_encoder(annotations)
        extra_info['time_encoder'] = extra_info['time_encoder'].end_timer()

        self.train_step += 1

        extra_info['time_backbone'] = CudaTimer.initiate_timer(active=self.enable_timing)
        outputs, features = self.backbone(images)
        extra_info['time_backbone'] = extra_info['time_backbone'].end_timer()

        extra_info['time_batch_loss'] = CudaTimer.initiate_timer(active=self.enable_timing)
        loss_dict = {
            'loss_jlabel': torch.zeros(1, device=device),
            'loss_joff': torch.zeros(1, device=device),
        }
        loss_dict.update(self.plane_classifier.initialize_loss(device))



        mask = targets['mask']
        for nstack, output in enumerate(outputs):
            jlabel_out, joff_out = self._get_output_dist(output)
            loss_dict['loss_jlabel'] += self.junction_label_loss(
                jlabel_out.flatten(start_dim=2),
                targets['jlabel'].flatten(start_dim=1),
                # jlabel_out.flatten(start_dim=2)[b_sample_idx, :, e_sample_idx],
                # jlabel_target_flat[b_sample_idx, e_sample_idx]
                )
            loss_dict['loss_joff'] += sigmoid_l1_loss(joff_out, targets['joff'], -0.5, targets['jloc'])

        jlabel_out, joff_out = self._get_output_dist(outputs[0])
        jlabel_prob = jlabel_out.softmax(1)
        jlabel = jlabel_prob.argmax(1)
        jloc_pred = 1-jlabel_prob[:,0,None]
        joff_pred= joff_out.sigmoid() - 0.5

        # Line batch loss
        self.line_classifier.forward_train_batch(outputs, features, targets)
        # Plane batch loss
        self.plane_classifier.forward_train_batch(features)

        extra_info['time_batch_loss'] = extra_info['time_batch_loss'].end_timer()

        batch_size = len(annotations)
        for i, (meta,ann) in enumerate(zip(metas, annotations)):
            N = meta['junc'].size(0)

            juncs_pred, juncs_valid_score, flat_index = get_junctions(non_maximum_suppression(jloc_pred[i]),joff_pred[i], topk=min(N*2+2,self.n_dyn_junc))

            t = CudaTimer.initiate_timer(active=self.enable_timing)
            line_output = self.line_classifier.forward_train_image(i, juncs_pred, meta)
            extra_info['time_line_classifier_total'] += t.end_timer()

            if line_output:
                t = CudaTimer.initiate_timer(active=self.enable_timing)
                self.plane_classifier.forward_train(i, juncs_pred, line_output, meta, ann, batch_size, img=images[i])
                extra_info['time_plane_classifier_total'] += t.end_timer()


        # Update with line detection losses
        line_loss_dict = self.line_classifier.reset_and_get_loss_dict()
        loss_dict.update(line_loss_dict)
        plane_loss_dict, plane_extra_info = self.plane_classifier.reset_and_get_loss_dict()
        loss_dict.update(plane_loss_dict)
        extra_info.update(plane_extra_info)

        return loss_dict, extra_info
