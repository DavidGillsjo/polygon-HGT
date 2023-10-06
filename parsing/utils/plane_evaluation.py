from parsing.dataset import build_test_dataset
from parsing.config import cfg
import os
import scipy
import random
import itertools
import os.path as osp
import matplotlib
from matplotlib.patches import Polygon
from descartes import PolygonPatch
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import shapely.geometry as sg
import shapely.ops as so
from shapely.strtree import STRtree
from tqdm import tqdm
from parsing.utils.visualization import ImagePlotter
from parsing.utils.comm import to_device
import seaborn as sns
import networkx as nx
import logging
import torch
import math
import cuspatial as cs
import cudf
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch.nn.functional import binary_cross_entropy, cross_entropy
from parsing.utils.logger import CudaTimer
from scipy.optimize import linear_sum_assignment

def centroid_from_polygon(poly_pos):
    if isinstance(poly_pos, torch.Tensor):
        centroid = torch_centroid_from_polygon(poly_pos)
        # sg_poly = sg.Polygon(poly_pos.to('cpu').tolist())
        # sg_centroid = sg_poly.centroid
        # sg_centroid = torch.tensor(sg_centroid.coords, dtype = torch.float32, device = centroid.device)
        # if sg_poly.area > 1 and not torch.allclose(sg_centroid, centroid, atol=0.1):
        #     print('sg_centroid',sg_centroid)
        #     print('centroid',centroid)
        #     print('Shapely area',sg_poly.area)
        #     X = poly_pos[:,0]
        #     X_i = X[:-1]
        #     X_i1 = X[1:]
        #     Y = poly_pos[:,1]
        #     Y_i = Y[:-1]
        #     Y_i1 = Y[1:]
        #     XY = (X_i*Y_i1 - X_i1*Y_i)
        #     A = XY.sum()/2.0
        #     print('area', A)
        #     print(sg_centroid - centroid)
        # assert sg_poly.area < 1 or torch.allclose(sg_centroid, centroid, atol=0.1)
    else:
        centroid = np_centroid_from_polygon(poly_pos)

    # sg_centroid = sg.Polygon(poly_pos).centroid
    # sg_centroid = np.array(sg_centroid)
    # assert np.all(np.isclose(sg_centroid, centroid))

    return centroid

def torch_centroid_from_polygon(poly_pos):
    X = poly_pos[:,0]
    X_i = X[:-1]
    X_i1 = X[1:]
    Y = poly_pos[:,1]
    Y_i = Y[:-1]
    Y_i1 = Y[1:]
    # if not np.all(np.isclose(poly_pos[0], poly_pos[-1])):
    #     print(poly_pos.shape)
    #     print(poly_pos)
    assert torch.all(torch.isclose(poly_pos[0], poly_pos[-1]))

    XY = (X_i*Y_i1 - X_i1*Y_i)
    A = XY.sum()/2.0
    if A.abs() > 0.1:
        centroid = torch.zeros_like(poly_pos[0])
        # X coordinate
        centroid[0] = XY.dot(X_i + X_i1) / (6*A + 1e-10)
        # Y coordinate
        centroid[1] = XY.dot(Y_i + Y_i1)  / (6*A + 1e-10)
    else:
        centroid = (poly_pos.max(dim=0)[0] + poly_pos.min(dim=0)[0])/2.0

    return centroid

def np_centroid_from_polygon(poly_pos):
    X = poly_pos[:,0]
    X_i = X[:-1]
    X_i1 = X[1:]
    Y = poly_pos[:,1]
    Y_i = Y[:-1]
    Y_i1 = Y[1:]
    # if not np.all(np.isclose(poly_pos[0], poly_pos[-1])):
    #     print(poly_pos.shape)
    #     print(poly_pos)
    assert np.all(np.isclose(poly_pos[0], poly_pos[-1]))

    XY = (X_i*Y_i1 - X_i1*Y_i)
    A = np.sum(XY)/2.0
    if np.abs(A) > 0.1:
        centroid = np.zeros_like(poly_pos[0])
        # X coordinate
        centroid[0] = np.dot((X_i + X_i1), XY) / (6*A + 1e-10)
        # Y coordinate
        centroid[1] = np.dot((Y_i + Y_i1), XY )  / (6*A + 1e-10)
    else:
        centroid = (poly_pos.max(axis=0) + poly_pos.min(axis=0))/2.0


    return centroid


def cu2pt(cu_frame):
    return from_dlpack(cu_frame.to_dlpack())

def pt2cu(torch_tensor):
    return cudf.from_dlpack(to_dlpack(torch_tensor))


class Polygon2MaskIoULossCUDA:
    def __init__(self, enable_timing = True, loss_weights = None):
        self.initialized = False
        self.enable_timing = enable_timing
        self.loss_weights = loss_weights

    @classmethod
    def create(cls, use_quadtree = True, max_depth = 2, enable_timing = True, loss_weights = None):
        if use_quadtree:
            return QuadTreePolygon2MaskIoULossCUDA(max_depth , enable_timing, loss_weights)
        else:
            return cls(enable_timing, loss_weights)


    def initialize(self, device):
        if self.initialized:
            return
        xy_range = torch.arange(0,128, device = device) + 0.5
        xy = torch.cartesian_prod(xy_range, xy_range)
        self.points = cs.GeoSeries.from_points_xy(pt2cu(xy.flatten().contiguous()))
        self.device = device

        if self.loss_weights is not None:
            self.loss_weights = self.loss_weights.to(device)


    def _planes_to_vectors(self, planes):
        assert len(planes) > 0
        points = []
        ring_offsets = [0]
        next_offset = 1
        for poly in planes:
            points.append(poly.T.flatten())
            ring_offsets.append(next_offset)
            next_offset += poly.size(0)
        

        points = torch.cat(points)
        ring_offsets = torch.tensor(ring_offsets, device = self.device) # Start Idx for each ring in the sequence
        poly_offsets = torch.arange(len(planes) + 1, device = self.device) # Assume each polygon has one ring.
        print(points.shape, "points")
        print(ring_offsets.shape, "ring_offsets")
        print(poly_offsets.shape, "poly_offsets")

        return cs.GeoSeries.from_polygons_xy(
            pt2cu(points),
            pt2cu(ring_offsets),
            pt2cu(poly_offsets),
            pt2cu(poly_offsets),
        )


    def polygons2masks(self, planes):
        self.initialize(planes[0].device)
        # Batches of 31 are supported
        start_idx = range(0,len(planes),31)
        print("nbr planes", len(planes))
        
        all_mask_tensor = []
        for s_idx in start_idx:
            e_idx = min(s_idx + 31, len(planes))
            cu_polygons = self._planes_to_vectors(planes[s_idx:e_idx])
            print("cu polygons", cu_polygons.shape)
            mask_frame = cs.point_in_polygon(self.points, cu_polygons)
            print("mask_frame", mask_frame.shape)
            mask_tensor = cu2pt(mask_frame).view(mask_frame.shape).to(torch.bool)
            print("mask etnsor", mask_tensor.shape)
            all_mask_tensor.append(mask_tensor)
        print(len(all_mask_tensor))
        mask_tensor = torch.cat(all_mask_tensor, dim=1)
        print(mask_tensor.shape)
        assert mask_tensor.size(1) == len(planes)
        return mask_tensor

    def _get_iou_and_matches(self, gt_planes, planes, pred_masks = None):
        # Compute masks
        gt_masks = self.polygons2masks(gt_planes)
        # plot_masks(junctions, edges, self.gt_masks, gt_planes,desc = 'gt_masks')
        gt_area = torch.sum(gt_masks, dim=0)

        if pred_masks is None:
            pred_masks = self.polygons2masks(planes)

        pred_area = torch.sum(pred_masks, dim=0)
        # print('Polygons test took',ctimer.end_timer())

        # Match and calculate loss
        # ctimer.start_timer()
        match = torch.zeros(pred_masks.size(1), dtype = int, device = self.device)
        iou = torch.zeros(pred_masks.size(1), dtype = float, device = self.device)

        for i in range(pred_masks.size(1)):
            intersection_mask = torch.logical_and(pred_masks[:,i,None], gt_masks)
            intersection_area = torch.sum(intersection_mask, dim=0)
            my_iou = intersection_area/torch.clamp(pred_area[i] + gt_area - intersection_area, min=1e-4)
            my_match = torch.argmax(my_iou)
            iou[i] = my_iou[my_match]
            match[i] = my_match


        return gt_masks, pred_masks, match, iou


    def evaluate_planes(self, gt_planes, gt_labels, planes, logits, junctions = None, edges = None, metrics= ['loss', 'iou'], pred_masks = None):
        metrics = set(metrics)

        gt_masks, pred_masks, match, iou = self._get_iou_and_matches(gt_planes, planes, pred_masks)
        # if torch.sum(pred_area) == 0:
        # desc = 'qt' if self.use_quadtree else ''
        # plot_masks(junctions, edges, pred_masks, planes,desc = f'pred_masks_{desc}')
        # plot_masks(junctions, edges, gt_masks, gt_planes,desc = f'gt_masks_{desc}')

        loss = torch.zeros(pred_masks.size(1), dtype = float, device = self.device)
        if 'loss' in metrics:
            # TODO: Remove loop
            for i, my_match  in enumerate(match):
                pred_logits_mask = torch.unsqueeze(logits[i,None].T*pred_masks[None,:,i], dim=0)
                gt_labels_mask = torch.unsqueeze(gt_masks[:,my_match]*gt_labels[my_match], dim=0)
                gt_labels_mask = gt_labels_mask.to(torch.long)
                ce_mask = gt_masks[:,my_match] | pred_masks[:,i]
                if torch.any(ce_mask):
                    loss[i] = cross_entropy(pred_logits_mask[:,:,ce_mask], gt_labels_mask[:,ce_mask], reduction = 'mean', weight = self.loss_weights )
                else:
                    loss[i] = 0

                # plot_cross_entropy_masks(pred_logits_mask, gt_labels_mask, ce_mask, desc=f'ce_{i}')

        return  match, iou, loss

    def _get_iou_and_matches_biparte(self, gt_planes, planes, pred_masks = None, gt_masks = None):
        # Compute masks
        if gt_masks is None:
            gt_masks = self.polygons2masks(gt_planes)

        gt_area = torch.sum(gt_masks, dim=0)

        if pred_masks is None:
            pred_masks = self.polygons2masks(planes)

        pred_area = torch.sum(pred_masks, dim=0)

        intersection_mask = torch.logical_and(pred_masks[:,:,None], gt_masks[:,None])
        intersection_area = torch.count_nonzero(intersection_mask, dim=0)
        iou = intersection_area.to(torch.float32)/torch.clamp(pred_area[:,None] + gt_area[None] - intersection_area, min=1e-4)
        pred_ind, gt_ind = linear_sum_assignment(iou.to('cpu').numpy(), maximize=True)
        pred_ind = torch.from_numpy(pred_ind)
        gt_ind = torch.from_numpy(gt_ind)
        match_iou = iou[pred_ind, gt_ind]

        return gt_masks, pred_masks, pred_ind, gt_ind, match_iou

    def evaluate_planes_biparte(self, gt_planes, gt_labels, planes, logits, junctions = None, edges = None, metrics= ['loss', 'iou'], pred_masks = None, gt_masks = None, iou_threshold = 0.5):
        metrics = set(metrics)

        gt_masks, pred_masks, pred_ind, gt_ind, iou = self._get_iou_and_matches_biparte(gt_planes, planes, pred_masks, gt_masks)
        # if torch.sum(pred_area) == 0:
        # desc = 'qt' if self.use_quadtree else ''
        # plot_masks(junctions, edges, pred_masks, planes,desc = f'pred_masks_{desc}')
        # plot_masks(junctions, edges, gt_masks, gt_planes,desc = f'gt_masks_{desc}')
        include_mask = iou > iou_threshold
        gt_ind = gt_ind[include_mask]
        pred_ind = pred_ind[include_mask]
        iou = iou[include_mask]


        loss = {}
        if 'loss' in metrics:
            # Initialize with background loss
            extended_gt = torch.zeros(pred_masks.size(1), dtype = torch.long, device = self.device)
            extended_gt[pred_ind] = gt_labels[gt_ind]
            loss['label'] = cross_entropy(logits, extended_gt, weight = self.loss_weights)
            loss['iou'] = torch.mean(1-iou).to(self.device) if iou.numel() > 0 else 0


        return pred_ind, gt_ind, iou, loss

    def _get_iou_and_matches_room_layout(self, gt_planes, planes, pred_masks, gt_masks):
        # Match from largest GT to smallest, only one match per GT allowed.
        if gt_masks is None:
            gt_masks = self.polygons2masks(gt_planes)

        gt_area = torch.sum(gt_masks, dim=0)

        if pred_masks is None:
            pred_masks = self.polygons2masks(planes)

        pred_area = torch.sum(pred_masks, dim=0)

        all_gt_idx = []
        all_pred_idx = []
        match_iou = []
        free_pred_mask = torch.ones(len(planes), dtype=torch.bool)
        match2original_idx = torch.arange(len(planes))

        intersection_mask = torch.logical_and(pred_masks[:,:,None], gt_masks[:,None])
        intersection_area = torch.count_nonzero(intersection_mask, dim=0)
        iou = intersection_area.to(torch.float32)/torch.clamp(pred_area[:,None] + gt_area[None] - intersection_area, min=1e-4)

        for gt_idx in torch.argsort(gt_area, descending = True):
            if not free_pred_mask.any():
                break
            matched_pred_idx = iou[free_pred_mask, gt_idx].argmax()
            pred_idx = match2original_idx[free_pred_mask][matched_pred_idx]
            free_pred_mask[pred_idx] = False
            all_gt_idx.append(gt_idx)
            all_pred_idx.append(pred_idx)
            match_iou.append(iou[pred_idx, gt_idx])

        pred_ind = torch.tensor(all_pred_idx, device=self.device, dtype=torch.long)
        gt_ind = torch.tensor(all_gt_idx, device=self.device, dtype=torch.long)
        match_iou = torch.tensor(match_iou, device=self.device, dtype=torch.float32)
        matching_intersection_mask = intersection_mask[:,all_pred_idx, all_gt_idx]
        if len(all_pred_idx) == 1:
            matching_intersection_mask = matching_intersection_mask[:,None]

        return gt_masks, pred_masks, pred_ind, gt_ind, match_iou, matching_intersection_mask


    def evaluate_planes_room_layout(self, gt_planes, planes, junctions = None, edges = None,pred_masks = None, gt_masks = None):

        gt_masks, pred_masks, pred_ind, gt_ind, iou, intesection_mask = self._get_iou_and_matches_room_layout(gt_planes, planes, pred_masks, gt_masks)

        metrics = {}
        nbr_gt = len(gt_planes)
        nbr_pred = len(planes)
        metrics['iou'] = 2/(nbr_gt + nbr_pred) * iou.sum()
        duplicate_mask = torch.zeros(gt_masks.size(0), dtype=torch.bool, device = pred_masks.device)
        for i in range(nbr_pred):
            if i not in pred_ind:
                duplicate_mask |= pred_masks[:,i]
        incorrect_mask_matched = ~intesection_mask.any(dim=1)
        incorrect_mask_all = incorrect_mask_matched  | duplicate_mask
        metrics['PE_matched'] =  incorrect_mask_matched.sum()/(gt_masks.size(0))
        metrics['PE_all'] =  incorrect_mask_all.sum()/(gt_masks.size(0))


        return pred_ind, gt_ind, metrics



class QuadTreePolygon2MaskIoULossCUDA(Polygon2MaskIoULossCUDA):
    def __init__(self, max_depth = 2, enable_timing = True, loss_weights = None):
        super().__init__(self, enable_timing, loss_weights)
        self.qt_max_depth = max_depth
        raise NotImplementedError('Quadtree not running correctly')

    def initialize(self, device):
        if self.initialized:
            return
        super().initialize(device)
        self.qt_pmin = 0
        self.qt_pmax = 129
        self.qt_min_size = 500
        self.qt_scale = 25
        self.qt_point_indices, self.qtree = cs.quadtree_on_points(self.points_x, self.points_y, self.qt_pmin,
                                                                  self.qt_pmax, self.qt_pmin, self.qt_pmax,
                                                                  self.qt_scale, self.qt_max_depth, self.qt_min_size)

        # Follow cuspatial point_in_polygon but work with qtree
        # https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial_indexing.html#cuspatial.point_in_polygon
    def polygons2masks(self, planes):
        self.initialize(planes[0].device)
        poly_dict = self._planes_to_vectors(planes)
        poly_points_x, poly_points_y, poly_offsets, poly_ring_offsets = poly_dict['poly_points_x'], poly_dict['poly_points_y'], poly_dict['poly_offsets'], poly_dict['poly_ring_offsets']
        poly_bb = cs.polygon_bounding_boxes(poly_offsets, poly_ring_offsets, poly_points_x, poly_points_y)
        poly_quad_pairs = cs.join_quadtree_and_bounding_boxes(self.qtree, poly_bb, self.qt_pmin,
                                                              self.qt_pmax, self.qt_pmin, self.qt_pmax,
                                                              self.qt_scale, self.qt_max_depth)
        pnp_result = cs.quadtree_point_in_polygon(poly_quad_pairs, self.qtree, self.qt_point_indices,
                                                  self.points_x, self.points_y, poly_offsets,
                                                  poly_ring_offsets, poly_points_x, poly_points_y)

        #Make mask
        mask_tensor = torch.zeros([self.points_x.size, poly_offsets.size], device = self.device, dtype=torch.bool)
        if not pnp_result.point_index.empty:
            point_index = cu2pt(pnp_result.point_index.astype(np.int32))
            polygon_index = cu2pt(pnp_result.polygon_index.astype(np.int32))
            mask_tensor[point_index, polygon_index] = True

        return mask_tensor

def generate_simple_geometry():
    device = 'cuda:0'
    junctions = 50*torch.tensor([(0,0),(1,0),(1,1),(0,1),(2,0)], device=device)
    edges = torch.tensor([
        (0,1),(1,2),(2,3),(3,0), #Square
        (1,4),(4,2), # Triangle right
        (0,2),(3,1) # split square x2
    ], dtype=torch.long, device=device)
    planes = [
        (0,1,2,0), #Square right
        (0,2,3,0), #Square left
        (0,1,3,0), #Square left 2
        (1,2,3,1), #Square right 2
        (1,4,2,1), #Triangle right
        (0,1,2,3,0) #Full square
    ]
    planes = [torch.tensor(p,dtype=torch.long, device=device) for p in planes]
    planes = [junctions[p] for p in planes]
    logits = torch.cat([
        -1e5*torch.ones([len(planes),1], device=device),
        1e5*torch.ones([len(planes),1], device=device)
        ], dim=1)
    gt_planes = [
        (0,1,4,2,3,0), #Full polygon
        (0,1,2,0), #Square right
    ]
    gt_planes = [torch.tensor(p, device=device) for p in gt_planes]
    gt_planes = [junctions[p] for p in gt_planes]
    gt_labels = torch.ones(len(gt_planes), dtype=torch.long, device=device)
    return junctions,edges, gt_planes, planes, logits, gt_labels


def plot_masks(junctions, edges, masks, planes, desc = 'mask'):
    def _plot_lines(ax, plane):
        ax.plot(junctions[:,0],junctions[:,1], 'b.')
        ax.plot(plane[:,0],plane[:,1], 'gs')
        for l in lines:
            ax.plot(l[:,0],l[:,1],'b')

    edges = edges.detach().cpu().numpy()
    junctions = junctions.detach().cpu().numpy()
    lines = junctions[edges,:]
    masks = masks.detach().cpu().numpy()
    planes = [p.detach().cpu().numpy() for p in planes]
    plt.figure()
    nbr_plots = masks.shape[1]
    row = int(np.ceil(np.sqrt(nbr_plots/1.6)))
    col = int(np.ceil(nbr_plots/row))

    colors = sns.color_palette("bright")

    for i in range(nbr_plots):
        ax = plt.subplot(row,col,i+1)
        plt.imshow(masks[:,i].reshape(128,128), alpha=0.8, vmin=0, vmax=1)
        _plot_lines(ax,planes[i])


    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_evaluation/{desc}.svg')

def plot_cross_entropy_masks(logit_img, target_img, ce_mask, desc = 'ce_mask'):
    ce = cross_entropy(logit_img, target_img, reduction = 'none' )
    ce_mask = ce_mask.cpu().squeeze().numpy()
    ce = ce.cpu().squeeze().numpy()
    logit_img = logit_img.cpu().squeeze().numpy()
    target_img = target_img.cpu().squeeze().numpy()
    plt.figure()
    nbr_plots = logit_img.shape[0] + 2
    row = int(np.ceil(np.sqrt(nbr_plots/1.6)))
    col = int(np.ceil(nbr_plots/row))

    colors = sns.color_palette("bright")
    ax = plt.subplot(row,col,1)
    plt.imshow(target_img.reshape(128,128))
    plt.title('Target')
    plt.colorbar()

    for i in range(1,nbr_plots-1):
        ax = plt.subplot(row,col,i+1)
        plt.imshow(logit_img[i-1,:].reshape(128,128))
        plt.title(f'Logit channel {i-1}')
        plt.colorbar()

    ax = plt.subplot(row,col,nbr_plots)
    ce_plot = ce_mask*ce
    ce_red = np.sum(ce_plot)/np.sum(ce_mask)
    plt.imshow(ce_plot.reshape(128,128))
    plt.title(f'Cross Entropy {ce_red:0.2e}')
    plt.colorbar()


    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_evaluation/{desc}.svg')


def plot_result(junctions, edges, gt_planes = [], gt_labels = [], planes = [], all_iou = [], predictions = [], all_loss = [], logits = [], desc = 'simple_geometry'):



    def _plot_polygon(p, c, ax):
        if isinstance(p, sg.Polygon):
            ax.add_patch(PolygonPatch(p,facecolor=c,edgecolor = c, alpha=0.5))
        else:
            ax.add_patch(Polygon(p,facecolor=c,edgecolor = c, alpha=0.5, closed=True))

    def _plot_lines(ax):
        ax.plot(junctions[:,0],junctions[:,1], 'b.')
        for l in lines:
            ax.plot(l[:,0],l[:,1],'b')

    if len(predictions) > 0:
        assert len(predictions) == len(planes)
        assert len(all_iou) == len(planes)

    edges = edges.cpu().numpy()
    junctions = junctions.cpu().numpy()
    lines = junctions[edges,:]
    planes = [p.cpu().numpy() for p in planes]
    gt_planes = [p.cpu().numpy() for p in gt_planes]

    plt.figure()
    nbr_plots = len(planes)
    row = int(np.ceil(np.sqrt(nbr_plots/1.6)))
    col = int(np.ceil(nbr_plots/row))

    colors = sns.color_palette("bright")

    axes = []
    for i, p in enumerate(planes):
        ax = plt.subplot(row,col,i+1)
        axes.append(ax)
        _plot_lines(ax)
        _plot_polygon(p,colors[1], ax)

    for i, (pred, iou, logit, loss) in enumerate(zip(predictions, all_iou, logits, all_loss)):
        ax = axes[i]
        score = torch.softmax(logit, dim=0)
        score = score[gt_labels[pred]]
        ax.set_title(f'IoU: {iou:0.2f}, S: {score:0.2f}, \nL: {loss:0.2f}', fontdict = {'fontsize': 8})
        p = gt_planes[pred]
        _plot_lines(ax)
        _plot_polygon(p,colors[0], ax)


    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_evaluation/{desc}.svg')


def prepare_data(ann, data_multplier = 1):
    #Take detections as both positive and negative edges
    junctions = ann['junctions']
    f_hw = 128.0
    sx = f_hw/ann['width']
    sy = f_hw/ann['height']
    junctions[:,0] = torch.clip(junctions[:,0]*sx, 0, f_hw-1e-4)
    junctions[:,1] = torch.clip(junctions[:,1]*sy, 0, f_hw-1e-4)
    edges = ann['edges_positive']
    gt_planes = [junctions[p['junction_idx']] for p in ann['planes']]
    gt_planes = gt_planes * data_multplier
    gt_labels = [p['semantic'] for p in ann['planes']]
    gt_labels = gt_labels * data_multplier
    pred_planes = [junctions[p['junction_idx']] for p in ann['planes_negative']]
    pred_planes = pred_planes * data_multplier
    logits = torch.randn([len(pred_planes)* data_multplier, 4], device = junctions.device)
    # scores = torch.ones(len(pred_planes), device = junctions.device)*0.9
    for p in gt_planes + pred_planes:
        assert torch.all(p[0] == p[-1])

    return junctions, edges, gt_planes, gt_labels, pred_planes, logits

if __name__ == '__main__':
    script_path = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Benchmark plane loss methods', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--nbr-samples', type=int, default = 1, help='Number of samples from dataset')
    parser.add_argument('-p', '--nbr-planes', type=int, default = math.inf, help='Number of planes to evaluate')
    parser.add_argument('-m', '--multiplier', type=int, default = 1, help='Extend the number of planes by this factor')

    args = parser.parse_args()
    cfg.merge_from_file(osp.join(script_path, '..', '..', 'config-files', 'Pred-simple-plane-S3D.yaml'))
    cfg.DATASETS.TEST = ("structured3D_perspective_planes_test_mini",)
    device = cfg.MODEL.DEVICE

    #Supress shapely logger
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    p2mask_notree = Polygon2MaskIoULossCUDA.create(use_quadtree = False)
    p2mask_qtree = {f'QTree D{d}': Polygon2MaskIoULossCUDA.create(use_quadtree = True, max_depth=d) for d in range(2,3)}


    functions = {'Polygon2MaskIoULossCUDA': lambda junctions,edges, gt_planes, gt_labels, pred_planes, logits :
                  p2mask_notree.evaluate_planes(gt_planes,gt_labels, pred_planes, logits, junctions = junctions,edges = edges)}
    qt_f = {f'Polygon2MaskIoULossCUDA - {s}': lambda junctions,edges, gt_planes, gt_labels, pred_planes, logits :
            p2mask_class.evaluate_planes(gt_planes,gt_labels, pred_planes, logits, junctions = junctions,edges = edges)
            for s,p2mask_class in p2mask_qtree.items()}
    functions.update(qt_f)
    timings = {f:[] for f in functions}
    IoU_score = {f:[] for f in functions}
    loss = {f:[] for f in functions}

    junctions, edges, gt_planes, pred_planes, logits, gt_labels = generate_simple_geometry()
    # plot_result(junctions, edges, gt_planes=gt_planes,planes=planes, desc='gt')

    for name, f in functions.items():
    #     t = time.process_time()
        predictions,all_iou, all_loss = f(junctions,edges, gt_planes, gt_labels, pred_planes, logits)
        plot_result(junctions, edges,
                    gt_planes=gt_planes,
                    gt_labels=gt_labels,
                    planes=pred_planes,
                    desc=f'generated_{name}',
                    predictions =predictions,
                    all_iou = all_iou,
                    logits = logits,
                    all_loss = all_loss)
    #     timings[name].append(time.process_time() - t)
    #     plot_result(junctions, edges,planes, desc=name)
    #     print(f'{name} found {len(planes)} planes')
    #     print(planes)
    sys.exit()
    ctimer = CudaTimer(self.enable_timing)
    datasets = build_test_dataset(cfg)
    for name, dataset in datasets:
        for i, (images, annotations) in enumerate(tqdm(dataset)):

            if i >= args.nbr_samples:
                break

            annotations = to_device(annotations, device)
            ann = annotations[0]
            junctions, edges, gt_planes, gt_labels, pred_planes, logits = prepare_data(ann, args.multiplier)
            print(f'Number of junctions {junctions.size(0)}')
            print(f'Number of gt_planes {len(gt_planes)}')
            print(f'Number of pred_planes {len(pred_planes)}')
            ann_filename = osp.splitext(ann['filename'])[0]

            for name, f in functions.items():
                ctimer.start_timer()
                predictions,all_iou, all_loss = f(junctions,edges, gt_planes, gt_labels, pred_planes, logits)
                timings[name].append(ctimer.end_timer())
                IoU_score[name] += all_iou
                loss[name] += all_loss
                # plot_result(junctions, edges,
                #             gt_planes=gt_planes,
                #             gt_labels=gt_labels,
                #             planes=pred_planes,
                #             desc=f'{ann_filename}_{name}',
                #             predictions =predictions,
                #             all_iou = all_iou,
                #             logits = logits,
                #             all_loss = all_loss)
    #
    # print(f'Evaluate {args.nbr_planes} planes over {args.nbr_samples} sample images')
    all_durations = []
    for method, durations in timings.items():
        print('=====================')
        print(method)
        print('---------------------')
        print('Median:', np.median(durations))
        print('Min:', np.min(durations))
        print('Max:', np.max(durations))
        print('')
        all_durations.append(durations)
    #
    plt.figure()
    # plt.subplot(2,1,1)
    # plt.bar(range(len(all_plane_counts)),all_plane_counts)
    # plt.ylabel('# planes generated')
    # plt.gca().tick_params(labelbottom=False)
    # plt.title(f'Generate {args.nbr_planes} planes for {args.nbr_samples} sample images')
    #
    #
    # plt.subplot(2,1,2)
    plt.boxplot(all_durations, labels = timings.keys(), showfliers=False)
    plt.xticks(rotation = 30, ha='right')
    plt.ylabel('Time [seconds]')


    plt.tight_layout()
    plt.savefig(f'/host_home/plots/plane_evaluation/summary_{args.nbr_planes}planes_{args.nbr_samples}samples.png')
    plt.close()
