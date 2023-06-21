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
from parsing.modules.gnn import WireframeGNNHead
import random

# Samples features along the line in given feature space.
# Then reduces number of sampling points with max pooling and finally stacks them.
class LinePooling(nn.Module):
    def __init__(self, n_sampling_points, n_final_points, feature_depth):
        super().__init__()
        self.n_sampling_points = n_sampling_points
        self.n_final_points = n_final_points
        self.feature_depth  = feature_depth
        self.register_buffer('tspan', torch.linspace(0, 1, n_sampling_points)[None,None,:])
        self.pool1d = nn.MaxPool1d(n_sampling_points//n_final_points, n_sampling_points//n_final_points)

    def forward(self, features_per_image, lines_per_im):
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)
              + features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)
              + features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)
              + features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128,-1,self.n_sampling_points)
        ).permute(1,0,2)

        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_final_points*self.feature_depth)

        # features_per_line = self.fc2(features_per_line)

        return features_per_line

class LineClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_pts0     = cfg.MODEL.PARSING_HEAD.N_PTS0
        self.n_pts1     = cfg.MODEL.PARSING_HEAD.N_PTS1
        self.dim_loi    = cfg.MODEL.PARSING_HEAD.DIM_LOI
        self.dim_fc     = cfg.MODEL.PARSING_HEAD.DIM_FC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.max_distance = cfg.MODEL.PARSING_HEAD.MAX_DISTANCE
        if self.max_distance <= 0:
            self.max_distance = float('inf')
        self.use_residual = cfg.MODEL.PARSING_HEAD.USE_RESIDUAL
        self.use_gt_lines = cfg.MODEL.USE_GT_LINES
        self.output_idx = np.cumsum([0] + [h[0] for h in cfg.MODEL.HEAD_SIZE])
        self.nbr_line_labels = len(cfg.MODEL.LINE_LABELS)
        # label_mapper = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        # self.nbr_line_labels = label_mapper.nbr_line_labels()

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:])

        if getattr(cfg.MODEL, 'LINE_LOSS_WEIGHTS', None):
            line_loss = torch.tensor(cfg.MODEL.LINE_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            line_loss = None
        self.loss = nn.CrossEntropyLoss(reduction='none', weight = line_loss)

        # Conv layer reduce feature depth
        self.fc1 = nn.Sequential(
            nn.Conv2d(256, self.dim_loi, 1),
            # nn.BatchNorm2d(self.dim_loi),
            nn.ReLU(inplace=True)
            )

        # Interpolation over line features and pooling. Works on reduces layer of depth dim_loi.
        self.pooling = LinePooling(self.n_pts0, self.n_pts1, self.dim_loi)

        line_bias = getattr(cfg.MODEL, 'LINE_CLASS_BIAS', None)
        if line_bias:
            last_fc.bias.weight = torch.tensor(line_bias, dtype=torch.float32)

        # FC layer for classification based on pooled features
        last_fc = nn.Linear(self.dim_fc, self.nbr_line_labels)
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
            )

        if cfg.MODEL.PARSING_HEAD.FREEZE:
            self._freeze()

    def _freeze(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = False


    def _get_output_dist(self, output):
        md_out = output[:,self.output_idx[0]:self.output_idx[1]]
        dis_out = output[:,self.output_idx[1]:self.output_idx[2]]
        res_out = output[:,self.output_idx[2]:self.output_idx[3]]
        return md_out, dis_out, res_out


    # Always batch size 1, so we can do all in one take
    def forward_test(self, outputs, features, juncs_pred, annotations = None):
        device = features.device

        extra_info = {
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }


        loi_features = self.fc1(features)
        md_out, dis_out, res_out = self._get_output_dist(outputs[0])
        md_pred = md_out.sigmoid()
        dis_pred = dis_out.sigmoid()
        res_pred = res_out.sigmoid()

        batch_size = md_pred.size(0)
        assert batch_size == 1


        extra_info['time_proposal'] = time.time()
        if self.use_gt_lines:
            ann = annotations[0]
            junctions = ann['junctions']
            junctions[:,0] *= 128/float(ann['width'])
            junctions[:,1] *= 128/float(ann['height'])
            edges_positive = ann['edges_positive']
            lines_pred = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1).to(device)
        elif self.use_residual:
            lines_pred = self.proposal_lines_new(md_pred[0],dis_pred[0],res_pred[0]).view(-1,4)
        else:
            lines_pred = self.proposal_lines_new(md_pred[0], dis_pred[0], None).view(-1, 4)


        extra_info['time_proposal'] = time.time() - extra_info['time_proposal']
        extra_info['time_matching'] = time.time()
        if juncs_pred.size(0) > 1:
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:,2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

            # iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)# * (dis_junc_to_end1< 10*10)*(dis_junc_to_end2<10*10)  # *(dis_junc_to_end2<100)
            iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)*(dis_junc_to_end1< self.max_distance**2)*(dis_junc_to_end2<self.max_distance**2)
        else:
            iskeep = torch.zeros(1, dtype=torch.bool)

        some_lines_valid = iskeep.count_nonzero() > 0
        if some_lines_valid:
            idx_lines_for_junctions = torch.unique(
                torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1),
                dim=0)

            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)

            extra_info['time_matching'] = time.time() - extra_info['time_matching']

            pooled_line_features = self.pooling(loi_features[0],lines_adjusted)

            # Filter lines
            line_logits = self.fc2(pooled_line_features)

            scores = line_logits.softmax(1)
            # TODO: Why is this done? And why not also filter the junctions?
            lines_score_valid = 1-scores[:,0]
            valid_mask = lines_score_valid > 0.01
            lines_final = lines_adjusted[valid_mask]
            pooled_line_features = pooled_line_features[valid_mask]
            line_logits = line_logits[valid_mask]


            # TODO: Supply edges for the junctions?
            unique_j_idx, l2j_idx = idx_lines_for_junctions[valid_mask].unique(return_inverse=True)

            extra_info['time_verification'] = time.time()
            scores = line_logits.softmax(1)
            lines_score_valid = 1-scores[:,0]
            lines_label = scores[:,1:].argmax(1) + 1
            lines_score_label = torch.gather(scores, 1, lines_label.unsqueeze(1)).squeeze(1)

            extra_info['time_verification'] = time.time() - extra_info['time_verification']
        else:
            extra_info['time_matching'] = time.time() - extra_info['time_matching']
            extra_info['time_verification'] = 0

        if annotations:
            width = annotations[0]['width']
            height = annotations[0]['height']
        else:
            width = images.size(3)
            height = images.size(2)

        sx = width/features.size(3)
        sy = height/features.size(2)

        lines_pred[:,0] *= sx
        lines_pred[:,1] *= sy
        lines_pred[:,2] *= sx
        lines_pred[:,3] *= sy
        extra_info['lines_prior_ver'] = lines_pred
        if some_lines_valid:
            lines_adjusted[:,0] *= sx
            lines_adjusted[:,1] *= sy
            lines_adjusted[:,2] *= sx
            lines_adjusted[:,3] *= sy
            extra_info['lines_prior_scoring'] = lines_adjusted
        else:
            extra_info['lines_prior_scoring'] = None


        output = {}

        if some_lines_valid:
            lines_final[:,0] *= sx
            lines_final[:,1] *= sy
            lines_final[:,2] *= sx
            lines_final[:,3] *= sy

            extra_info['valid_junction_idx'] = unique_j_idx

            output.update({
                'lines_pred': lines_final,
                'lines_label': lines_label,
                'lines_valid_score': lines_score_valid,
                'lines_label_score': lines_score_label,
                'lines_score': scores,
                'edges_pred': l2j_idx,
                'num_proposals': lines_adjusted.size(0),
            })
        else:
            output.update({
                'lines_pred': torch.tensor([]),
                'num_proposals': 0
                })

        return output, extra_info, loi_features[0]

    # Handle batch losses, store state variables for forward_train_image
    def forward_train_batch(self, outputs, features, targets):
        device = features.device

        self.loss_dict = {
            'loss_md': torch.zeros(1, device=device),
            'loss_dis': torch.zeros(1, device=device),
            'loss_res': torch.zeros(1, device=device),
            'loss_pos': torch.zeros(1, device=device),
            'loss_neg': torch.zeros(1, device=device),
        }


        mask = targets['mask']
        for nstack, output in enumerate(outputs):
            md_out, dis_out, res_out  = self._get_output_dist(output)
            loss_map = torch.mean(F.l1_loss(md_out.sigmoid(), targets['md'],reduction='none'),dim=1,keepdim=True)
            self.loss_dict['loss_md']  += torch.mean(loss_map*mask) / torch.mean(mask)
            loss_map = F.l1_loss(dis_out.sigmoid(), targets['dis'], reduction='none')
            self.loss_dict['loss_dis'] += torch.mean(loss_map*mask) /torch.mean(mask)
            loss_residual_map = F.l1_loss(res_out.sigmoid(), loss_map, reduction='none')
            self.loss_dict['loss_res'] += torch.mean(loss_residual_map*mask)/torch.mean(mask)

        self.loi_features = self.fc1(features)
        md_out, dis_out, res_out = self._get_output_dist(outputs[0])
        self.md_pred = md_out.sigmoid()
        self.dis_pred = dis_out.sigmoid()
        self.res_pred = res_out.sigmoid()

    def reset_and_get_loss_dict(self):
        # Free some memory
        del self.loi_features
        del self.md_pred
        del self.dis_pred
        del self.res_pred
        return self.loss_dict

    # Handle line detection, called for every image.
    # img_idx is the index for the image in the batch, so we can correctly use our state variables
    def forward_train_image(self, img_idx, juncs_pred, meta):
        device = juncs_pred.device
        batch_size = self.md_pred.size(0)

        # No junctions, just add static training examples
        if juncs_pred.size(0) < 2:
            logits = self.pooling(self.loi_features[img_idx],meta['lpre'])
            loss_ = self.loss(logits, meta['lpre_label'])

            loss_positive = loss_[meta['lpre_label']>0].mean()
            loss_negative = loss_[meta['lpre_label']==0].mean()

            self.loss_dict['loss_pos'] += loss_positive/batch_size
            self.loss_dict['loss_neg'] += loss_negative/batch_size
            return None

        junction_gt = meta['junc']
        N = junction_gt.size(0)

        md_pred_per_im = self.md_pred[img_idx]
        dis_pred_per_im = self.dis_pred[img_idx]
        res_pred_per_im = self.res_pred[img_idx]
        loi_features_per_img = self.loi_features[img_idx]

        if not self.use_gt_lines:
            lines_pred = []
            if self.use_residual:
                for scale in [-1.0,0.0,1.0]:
                    _ = self.proposal_lines(md_pred_per_im, dis_pred_per_im+scale*res_pred_per_im).view(-1, 4)
                    lines_pred.append(_)
            else:
                lines_pred.append(self.proposal_lines(md_pred_per_im, dis_pred_per_im).view(-1, 4))
            lines_pred = torch.cat(lines_pred)
        else:
            lines_pred = meta['lines']

        dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
        dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
        iskeep = idx_junc_to_end_min<idx_junc_to_end_max
        idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1).unique(dim=0)
        # idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
        # idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
        lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)

        cost_, match_ = torch.sum((juncs_pred-junction_gt[:,None])**2,dim=-1).min(0)
        match_[cost_>1.5*1.5] = N
        Lpos = meta['Lpos']
        labels = Lpos[match_[idx_lines_for_junctions[:,0]],match_[idx_lines_for_junctions[:,1]]]

        #TODO: Try weighting the loss depending on how close the match is. Similair to IoU.

        iskeep = torch.zeros_like(labels, dtype= torch.bool)
        cdx = labels.nonzero().flatten()

        if len(cdx) > self.n_dyn_posl:
            perm = torch.randperm(len(cdx),device=device)[:self.n_dyn_posl]
            cdx = cdx[perm]

        iskeep[cdx] = True

        if self.n_dyn_negl >0 :
            cdx = (labels==0).nonzero().flatten()
            if len(cdx) > self.n_dyn_negl:
                perm = torch.randperm(len(cdx), device=device)[:self.n_dyn_negl]
                cdx = cdx[perm]
            iskeep[cdx] = True

        all_lines = torch.cat((lines_adjusted,meta['lpre']))
        all_labels = torch.cat((labels,meta['lpre_label']))

        pooled_line_features = self.pooling(self.loi_features[img_idx],all_lines)
        line_logits = self.fc2(pooled_line_features)
        all_iskeep = torch.cat((iskeep,torch.ones_like(meta['lpre_label'], dtype= torch.bool)))

        selected_labels = all_labels[all_iskeep]

        selected_logits = line_logits[all_iskeep]
        loss = self.loss(selected_logits, selected_labels)

        self.loss_dict['loss_pos'] += loss[selected_labels>0].mean()/batch_size
        self.loss_dict['loss_neg'] += loss[selected_labels==0].mean()/batch_size

        output = {
            'edges_pred': idx_lines_for_junctions,
            'lines_pred': lines_adjusted,
            'features': self.loi_features[img_idx],
            'lines_logits': line_logits[:lines_adjusted.size(0)]
        }
        return output


# TODO: Do we need this?
    def proposal_lines(self, md_maps, dis_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        x_standard = torch.ones_like(cs_st)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated =  (cs_md - ss_md*y_st)*dis_maps[0]*scale
        y_st_rotated =  (ss_md + cs_md*y_st)*dis_maps[0]*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)*dis_maps[0]*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*dis_maps[0]*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,0))

        return  lines#, normals

    def proposal_lines_new(self, md_maps, dis_maps, residual_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        sign_pad     = torch.tensor([-1,0,1],device=device,dtype=torch.float32).reshape(3,1,1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1,1,1))
        else:
            dis_maps_new = dis_maps.repeat((3,1,1))+sign_pad*residual_maps.repeat((3,1,1))
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated = (cs_md-ss_md*y_st)[None]*dis_maps_new*scale
        y_st_rotated =  (ss_md + cs_md*y_st)[None]*dis_maps_new*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)[None]*dis_maps_new*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)[None]*dis_maps_new*scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,3,0))

        # normals = torch.stack((cs_md,ss_md)).permute((1,2,0))

        return  lines#, normals
