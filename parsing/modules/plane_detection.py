import torch
from torch import nn
# from epnet.structures.linelist_ops import linesegment_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from parsing.utils.labels import LabelMapper
from parsing.utils.plane_evaluation import Polygon2MaskIoULossCUDA, centroid_from_polygon
from parsing.utils.plane_generation import CycleBasisGeneration
from parsing.modules.line_detection import LinePooling
from parsing.utils.logger import CudaTimer
import random


class PlaneClassifier(nn.Module):
    subclasses = {}

    @classmethod
    def register_subclass(cls, strategy):
        def decorator(subclass):
            cls.subclasses[strategy] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, cfg):
        strategy = cfg.MODEL.PLANE_HEAD.STRATEGY
        if strategy not in cls.subclasses:
            raise ValueError('Bad strategy {}'.format(strategy))

        return cls.subclasses[strategy](cfg)

    def __init__(self, cfg):
        super().__init__()

        self.n_dyn_p         = cfg.MODEL.PLANE_HEAD.N_DYN_PLANES
        self.n_static_posp      = cfg.MODEL.PLANE_HEAD.N_STATIC_POS_PLANES
        self.n_static_negp      = cfg.MODEL.PLANE_HEAD.N_STATIC_NEG_PLANES
        self.dim_line_pooling   = cfg.MODEL.PLANE_HEAD.DIM_LINE_POOL
        # The smallest polygon has 3 lines, so we cannot have more outputs than that.
        assert self.dim_line_pooling <= 3
        self.n_pts0             = cfg.MODEL.PLANE_HEAD.N_PTS0
        self.n_pts1             = cfg.MODEL.PLANE_HEAD.N_PTS1
        self.dim_fc             = cfg.MODEL.PLANE_HEAD.DIM_FC
        self.dim_loi            = cfg.MODEL.PARSING_HEAD.DIM_LOI #LOI is the same since we use features from the line classifier

        self.topk_edges = cfg.MODEL.PLANE_HEAD.TOPK_EDGES
        self.nbr_labels = len(cfg.MODEL.PLANE_LABELS)
        self.edge_valid_score_threshold = cfg.MODEL.PLANE_HEAD.EDGE_VALID_SCORE_THRESHOLD
        self.topk_output_planes = cfg.MODEL.PLANE_HEAD.TOPK_OUTPUT_PLANES
        self.plane_nms_threshold = cfg.MODEL.PLANE_HEAD.NMS_IOU_THRESHOLD


        if getattr(cfg.MODEL.PLANE_HEAD, 'LOSS_WEIGHTS', None):
            loss_weights = torch.tensor(cfg.MODEL.PLANE_HEAD.LOSS_WEIGHTS, dtype=torch.float32)
        else:
            loss_weights = None

        self._setup_sampling(cfg)

        # Conv layer reduce feature depth
        self.fc1 = nn.Sequential(
            nn.Conv2d(256, self.dim_loi, 1),
            # nn.BatchNorm2d(self.dim_loi),
            nn.ReLU(inplace=True)
            )

        self.loss_initialized = False
        self.enable_timing = cfg.MODEL.ENABLE_TIMING
        self.eval = Polygon2MaskIoULossCUDA.create(use_quadtree = cfg.MODEL.PLANE_HEAD.USE_QUADTREE,
                                                   enable_timing = self.enable_timing,
                                                   loss_weights = loss_weights)


    @classmethod
    def initialize_loss(cls, device):
        loss_dict = {
            'loss_plane_pos': torch.zeros(1, device=device),
            'loss_plane_neg': torch.zeros(1, device=device)
            }
        return loss_dict

    def initialize(self, device):
        self.loss_dict = self.initialize_loss(device)
        self.extra_info = {
            'time_plane_generation': 0.0,
            'time_plane_read_ann': 0.0,
            'time_plane_sample_logits': 0.0,
            'time_plane_calculate_loss': 0.0,
        }
        self.loss_initialized = True

    def reset_and_get_loss_dict(self):
        self.loss_initialized = False

        # Free memory
        del self.plane_features

        return self.loss_dict, self.extra_info

    def _plane_ann_to_poly(self, ann):
        junctions = ann['junctions']
        gt_planes_pos = [junctions[p['junction_idx']] for p in ann['planes']]
        gt_labels = torch.tensor([p['semantic'] for p in ann['planes']], dtype=torch.long)

        if len(ann['planes']) > self.n_static_posp:
            pos_planes = random.sample(ann['planes'], self.n_static_posp)
        else:
            pos_planes = ann['planes']

        static_planes_pos = [junctions[p['junction_idx']] for p in pos_planes]
        static_planes_edge_idx = [p['edge_idx'] for p in pos_planes]

        if len(ann['planes_negative']) > self.n_static_negp:
            neg_planes = random.sample(ann['planes_negative'], self.n_static_negp)
        else:
            neg_planes = ann['planes_negative']

        static_planes_pos += [junctions[p['junction_idx']] for p in neg_planes]
        static_planes_edge_idx += [p['edge_idx'] for p in neg_planes]



        return gt_planes_pos, gt_labels, static_planes_pos, static_planes_edge_idx


    def _pool_line_features(self, features, plane_edge_idx, lines):

        all_edge_idx, inverse_index = torch.unique(torch.cat(plane_edge_idx), return_inverse = True)
        lines_present = lines[all_edge_idx]
        line_features = self.line_sampling_pooling(features, lines_present)
        inversed_line_features = line_features[inverse_index]
        features_same_size = []
        p_start = 0
        for p in plane_edge_idx:
            p_end = p_start + len(p)
            pooled_plane_feature = self.pool_plane_lines(inversed_line_features[p_start:p_end].T).view([1,-1])
            features_same_size.append(pooled_plane_feature)
            p_start = p_end

        features_same_size = torch.cat(features_same_size, dim=0)

        return features_same_size

    def forward_test(self, juncs_pred, line_output, features, img=None):

        extra_info = {}
        if line_output['lines_pred'].numel() == 0:
            return {'planes_pred': []}, extra_info

        device = juncs_pred.device
        ctimer = CudaTimer(self.enable_timing)


        # Filter lines based on score
        score_mask = line_output['lines_valid_score'] > self.edge_valid_score_threshold
        lines_pred = line_output['lines_pred'][score_mask]
        edges_pred = line_output['edges_pred'][score_mask]
        lines_valid_score = line_output['lines_valid_score'][score_mask]

        # Filter line output based on valid score to make sure there are not too many.
        if self.topk_edges < lines_pred.size(0):
            _, topk_indices = torch.topk(lines_valid_score, self.topk_edges, sorted=False)
            lines_pred = lines_pred[topk_indices]
            edges_pred = edges_pred[topk_indices]

        # Sample planes ALL planes
        ctimer.start_timer()
        cb = CycleBasisGeneration(edges_pred, juncs_pred, enable_timing = self.enable_timing)
        hyp_planes_node_idx, hyp_planes_edge_idx = cb.generate_planes(return_edge_idx = True)
        hyp_planes_pos = [juncs_pred[p] for p in hyp_planes_node_idx]
        extra_info['time_plane_generation'] = ctimer.end_timer()

        # Get timings
        extra_info.update(cb.get_timings())

        if not hyp_planes_edge_idx:
            return {'planes_pred': []}, extra_info

        # Get features forward to logits
        ctimer.start_timer()
        feature_cube = self.fc1(features)[0] # Single batch in testing
        pred_logits, planes_mask = self._classify_planes(feature_cube, plane_pos = hyp_planes_pos, plane_edge_idx = hyp_planes_edge_idx, lines_pos = lines_pred)
        extra_info['time_plane_sample_logits'] = ctimer.end_timer()

        planes_scores = pred_logits.softmax(1)
        planes_label = planes_scores.argmax(1)
        planes_score_label = torch.gather(planes_scores, 1, planes_label.unsqueeze(1)).squeeze(1)
        planes_score_valid = 1-planes_scores[:,0]

        if self.plane_nms_threshold > 0:
            keep_idx = self._nms_iou_simple(planes_mask, planes_score_valid)
            planes_scores = planes_scores[keep_idx]
            planes_score_valid = planes_score_valid[keep_idx]
            planes_score_label = planes_score_label[keep_idx]
            planes_label = planes_label[keep_idx]
            hyp_planes_node_idx = [hyp_planes_node_idx[i] for i in keep_idx]


        if self.topk_output_planes < len(hyp_planes_node_idx):
            _, topk_indices = torch.topk(planes_score_valid, self.topk_output_planes, sorted=False)
            planes_scores = planes_scores[topk_indices]
            planes_score_valid = planes_score_valid[topk_indices]
            planes_score_label = planes_score_label[topk_indices]
            planes_label = planes_label[topk_indices]
            hyp_planes_node_idx = [hyp_planes_node_idx[i] for i in topk_indices]


        output = {
            'planes_score': planes_scores,
            'planes_label': planes_label,
            'planes_label_score': planes_score_label,
            'planes_valid_score': planes_score_valid,
            'planes_pred': hyp_planes_node_idx
        }

        return output, extra_info

    def forward_train_batch(self, features):
        self.plane_features = self.fc1(features)

    def forward_train(self, img_idx, juncs_pred, line_output, meta, ann, batch_size, img=None, batch_idx = None):
        device = juncs_pred.device
        ctimer = CudaTimer(self.enable_timing)

        if not self.loss_initialized:
            self.initialize(device)

        # Get annotated planes
        ctimer.start_timer()
        gt_planes_pos, gt_labels, static_planes_pos, static_planes_edge_idx = self._plane_ann_to_poly(ann)
        self.extra_info['time_plane_read_ann'] += ctimer.end_timer()

        # Filter line output based on valid score
        if self.topk_edges < line_output['lines_logits'].size(0):
            scores = line_output['lines_logits'].softmax(1)
            valid_score = 1-scores[:,0]
            _, topk_indices = torch.topk(valid_score, self.topk_edges, sorted=False)
            for k in ['edges_pred','lines_pred']:
                line_output[k] = line_output[k][topk_indices]

        # Sample planes
        ctimer.start_timer()
        cb = CycleBasisGeneration(line_output['edges_pred'], juncs_pred, enable_timing = self.enable_timing)
        hyp_planes_node_idx, hyp_planes_edge_idx = cb.generate_random_planes(number_of_planes = self.n_dyn_p, return_edge_idx = True, method='shortest', timeout = 1)
        hyp_planes_pos = [juncs_pred[p] for p in hyp_planes_node_idx]
        self.extra_info['time_plane_generation'] += ctimer.end_timer()

        # Accumulate timings
        cb_time_info = cb.get_timings()
        if next(iter(cb_time_info)) in self.extra_info:
            for k,v in cb_time_info.items():
                self.extra_info[k] += v
        else:
            self.extra_info.update(cb_time_info)



        # Get features forward to logits
        ctimer.start_timer()
        feature_cube = self.plane_features[img_idx]
        if hyp_planes_edge_idx:
            pred_logits, pred_masks = self._classify_planes(feature_cube, plane_pos = hyp_planes_pos, plane_edge_idx = hyp_planes_edge_idx, lines_pos = line_output['lines_pred'])

        static_lines_exist = len(meta.get('lines', [])) > 0
        if static_lines_exist:
            static_logits, static_masks = self._classify_planes(feature_cube, plane_pos = static_planes_pos, plane_edge_idx = static_planes_edge_idx, lines_pos = meta['lines'])
        self.extra_info['time_plane_sample_logits'] += ctimer.end_timer()


        # Merge planes
        if hyp_planes_edge_idx and static_lines_exist:
            all_planes_pos = hyp_planes_pos + static_planes_pos
            all_logits = torch.cat([pred_logits, static_logits], dim=0)
            all_masks = None if pred_masks is None else torch.cat([pred_masks, static_masks], dim=1)
        elif hyp_planes_edge_idx:
            all_planes_pos = hyp_planes_pos
            all_logits = pred_logits
            all_masks = pred_masks
        elif static_lines_exist:
            all_planes_pos = static_planes_pos
            all_logits = static_logits
            all_masks = static_masks
        else:
            return

        # Calculate loss
        ctimer.start_timer()
        match, all_iou, all_loss = self.eval.evaluate_planes(gt_planes_pos, gt_labels, all_planes_pos, all_logits, metrics=['loss', 'iou'], pred_masks = all_masks)
        # match, all_iou, all_loss = self.eval.evaluate_planes(gt_planes_pos,gt_labels, hyp_planes_pos, pred_logits, junctions = juncs_pred, edges = line_output['edges_pred'])
        # match, all_iou, all_loss = self.eval.evaluate_planes(gt_planes_pos,gt_labels, static_planes_pos, static_logits)

        pos_mask = all_iou > 0.5
        if torch.any(pos_mask):
            self.loss_dict['loss_plane_pos'] += all_loss[pos_mask].mean()/batch_size
        if torch.any(~pos_mask):
            self.loss_dict['loss_plane_neg'] += all_loss[~pos_mask].mean()/batch_size

        assert torch.isfinite(self.loss_dict['loss_plane_pos'])
        assert torch.isfinite(self.loss_dict['loss_plane_neg'])
        self.extra_info['time_plane_calculate_loss'] += ctimer.end_timer()


    def _nms_iou_label(self, planes_mask, planes_score_label):
        pass

    def _nms_iou_simple(self, planes_mask, planes_max_score):
        sorted_idx = torch.argsort(planes_max_score, descending=True)

        if sorted_idx.numel() == 1:
            return sorted_idx[0,None]

        all_area = torch.sum(planes_mask, dim=0)
        keep_idx = [sorted_idx[0]]
        sorted_idx = sorted_idx[1:]
        # Loop until out of indices to check
        while True:
            query_idx = keep_idx[-1]

            intersection_mask = torch.logical_and(planes_mask[:,query_idx,None], planes_mask[:,sorted_idx])
            intersection_area = torch.sum(intersection_mask, dim=0)
            iou = intersection_area/torch.clamp(all_area[query_idx] + all_area[sorted_idx] - intersection_area, min=1e-4)

            sorted_idx = sorted_idx[iou < self.plane_nms_threshold]


            if sorted_idx.numel() > 1:
                # Loop again
                keep_idx.append(sorted_idx[0])
                sorted_idx = sorted_idx[1:]
            elif sorted_idx.numel() == 1:
                keep_idx.append(sorted_idx[0])
                break
            else:
                break


        return torch.tensor(keep_idx)

    def _setup_sampling(self, cfg):
        raise NotImplementedError('Use a subclass of this module')

    def _classify_planes(self, *args, **kwargs):
        raise NotImplementedError('Use a subclass of this module')

    def _sample_polygon_feature(self, feature_cube, plane_pos):
        # Generate masks for polygons
        plane_masks = self.eval.polygons2masks(plane_pos)

        # Initialize
        h,w = feature_cube.size(1), feature_cube.size(2)
        plane_masks_2d = plane_masks.reshape([h,w,-1])
        pooled_features = torch.zeros([plane_masks.size(1), 4*feature_cube.size(0)], device=feature_cube.device)
        all_centroids = torch.zeros([plane_masks.size(1), 2], device=feature_cube.device)

        # Generate features vector for each polygon.
        for k, p_pos in enumerate(plane_pos):
            p_mask = plane_masks_2d[:,:,k]
            # Find centroid pixel
            centroid = centroid_from_polygon(p_pos)
            all_centroids[k] = centroid
            centroid_pixel_x = centroid[0].floor().clamp(min=0, max=w-1).to(int)
            centroid_pixel_y = centroid[1].floor().clamp(min=0, max=h-1).to(int)

            # Create masks for quadrants given centroid as center
            quadrant_mask = torch.zeros([4,h,w], device = p_mask.device, dtype=torch.bool)
            quadrant_mask[0,:centroid_pixel_y, :centroid_pixel_x] = 1
            quadrant_mask[1, :centroid_pixel_y, centroid_pixel_x:] = 1
            quadrant_mask[2, centroid_pixel_y:, :centroid_pixel_x] = 1
            quadrant_mask[3, centroid_pixel_y:, centroid_pixel_x:] = 1

            # Create final quadrant mask based on polygon
            quadrant_pmask = quadrant_mask & p_mask

            # Include centroid if some quadrant is empty
            quadrant_pmask[:,centroid_pixel_y, centroid_pixel_x] |= ~quadrant_pmask.view(4,-1).any(dim=1)

            # Do max pooling on features in each quadrant and concatenate to a feature vector
            quadrant_features = torch.zeros([4,feature_cube.size(0)], device=feature_cube.device)
            for i in range(4):
                q_pool, _ = feature_cube[:,quadrant_pmask[i]].max(dim=1)
                quadrant_features[i] = q_pool
            pooled_features[k] = quadrant_features.flatten()

        return pooled_features, plane_masks, all_centroids


@PlaneClassifier.register_subclass('line_sampling')
class LineSamplingPlaneClassifier(PlaneClassifier):
    def _setup_sampling(self, cfg):
        # Line pooling operation from the line classifier.
        # Used to sample lines contained in the plane sampling.
        # Since lines are sampled during training it is not possible to re-use pooled features from the line classifier.
        self.line_sampling_pooling = LinePooling(self.n_pts0, self.n_pts1, self.dim_loi)

        # Pooling operation to reduce any the number of line features from 3+ to 3.
        # Needed since each polygon may consist of 3 lines or more.
        self.pool_plane_lines = torch.nn.AdaptiveMaxPool1d(self.dim_line_pooling)

        # FC Layers for classification
        last_fc = nn.Linear(self.dim_fc, self.nbr_labels)
        self.fc2 = nn.Sequential(
            nn.Linear(self.n_pts1 * self.dim_loi * self.dim_line_pooling, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
            )

    def _classify_planes(self, feature_cube, plane_edge_idx = None, plane_pos = None, lines_pos = None):
        plane_features = self._pool_line_features(feature_cube, plane_edge_idx, lines_pos)
        pred_logits = self.fc2(plane_features)
        return pred_logits, None


@PlaneClassifier.register_subclass('polygon_sampling')
class PolySamplingPlaneClassifier(PlaneClassifier):
    def _setup_sampling(self, cfg):
        # FC Layers for classification
        last_fc = nn.Linear(self.dim_fc, self.nbr_labels)
        self.fc2 = nn.Sequential(
            nn.Linear(4 * self.dim_loi, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
            )

    def _classify_planes(self, feature_cube, plane_edge_idx = None, plane_pos = None, lines_pos = None):

        pooled_features, plane_masks, _ = self._sample_polygon_feature(feature_cube, plane_pos)
        # Fully connected layer
        pred_logits = self.fc2(pooled_features)

        return pred_logits, plane_masks


@PlaneClassifier.register_subclass('dummy')
class DummyPlaneClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def initialize_loss(self, *args, **kwargs):
        return {}

    def forward_test(self, *args, **kwargs):
        return {'planes_pred': []}, {}

    def forward_train(self, *args, **kwargs):
        pass

    def forward_train_batch(self, *args, **kwargs):
        pass

    def reset_and_get_loss_dict(self, *args, **kwargs):
        return {}, {}
