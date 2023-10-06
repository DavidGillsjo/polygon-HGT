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
from parsing.modules.plane_detection import PlaneClassifier
from parsing.utils.loss import sigmoid_l1_loss, scaled_sigmoid
import random
from scipy.optimize import linear_sum_assignment
import shapely.geometry as sg

# For HGT
import torch_geometric.transforms as T
# from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
import networkx as nx

class HGT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_types, num_heads, num_layers):
        super().__init__()

        # Assuming this is node specific transforms to graph features.
        # Is linear enough?
        self.lin_dict = torch.nn.ModuleDict()
        for node_type, in_ch in in_channels.items():
            self.lin_dict[node_type] = Linear(in_ch, hidden_channels)

        self.out_lin_dict = torch.nn.ModuleDict()
        for node_type in in_channels.keys():
            self.out_lin_dict[node_type] = Linear(hidden_channels, out_channels)

        metadata = [
            in_channels,
            edge_types
        ]

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        x_dict = {
            node_type: self.out_lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        return x_dict

class MultiScalePlaneConv(nn.Module):
    def __init__(self, in_channels, out_channels, img_side_length, anchors_per_side):
        super().__init__()
        assert anchors_per_side > 1
        sections_per_side = anchors_per_side - 1
        stride = int((img_side_length-1)/sections_per_side)
        self.stride = stride
        self.img_side_length = img_side_length
        self.multi_scale_plane_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels//4, 1, stride=stride),
            nn.Conv2d(in_channels, out_channels//4, 3, padding=1, stride=stride),
            nn.Conv2d(in_channels, out_channels//4, 3, dilation=1, padding=2, stride=stride),
            nn.Conv2d(in_channels, out_channels//4, 3, dilation=2, padding=3, stride=stride),
        ])
        self.activation = nn.ReLU(inplace=True)

    def _get_stride_positions(self):
        return torch.arange(0, self.img_side_length, self.stride) + 0.5

    def get_grid_positions(self):
        single_dim_positions = self._get_stride_positions()
        X, Y = torch.meshgrid(single_dim_positions, single_dim_positions, indexing='xy')
        return X,Y

    def get_grid_positions_normalized(self):
        single_dim_positions_n = self._get_stride_positions()/self.img_side_length
        X, Y = torch.meshgrid(single_dim_positions_n, single_dim_positions_n, indexing='xy')
        return X,Y



    def forward(self, x):
        y = [conv(x) for conv in self.multi_scale_plane_conv]
        y = torch.cat(y, dim=1)
        return self.activation(y)

class ProposalGNN(nn.Module):
    def __init__(self, hgt_in_channels, hgt_hidden_channels, hgt_out_channels, hgt_edge_types, nbr_plane_labels, num_layers = 3, num_heads = 8):
        super().__init__()
        # Original model had hidden_channels = 256, num_heads = 8, num_layers = 3
        self.graph_model = HGT(in_channels=hgt_in_channels,
                               hidden_channels=hgt_hidden_channels,
                               out_channels=hgt_out_channels,
                               edge_types = hgt_edge_types,
                               num_heads=num_heads,
                               num_layers=num_layers)

        # Classifier layers
        self.pc_label_layers = nn.Sequential(
            nn.Linear(hgt_out_channels, hgt_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hgt_out_channels, nbr_plane_labels),
            nn.ReLU(inplace=True)
        )
        self.pc_offset_layers = nn.Sequential(
            nn.Linear(hgt_out_channels, hgt_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hgt_out_channels, 2),
            nn.ReLU(inplace=True)
        )

        # Initialize edge prediction weight
        w = torch.empty(hgt_out_channels, hgt_out_channels)
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        w += torch.eye(hgt_out_channels)
        self.plane_line_edge_layer = nn.Parameter(w)

    def forward(self, data):
        node_features = self.graph_model(data.x_dict, data.edge_index_dict)

        # Logits for centroid
        pc_label_logits = self.pc_label_layers(node_features['plane'])
        pc_offset_logits = self.pc_offset_layers(node_features['plane'])

        # Make edge predictions for which lines belongs to which plane
        plane_line_logits = node_features['plane'].matmul(self.plane_line_edge_layer).matmul(node_features['line'].T)

        return pc_label_logits, pc_offset_logits, plane_line_logits


class FinalGNN(nn.Module):
    def __init__(self, hgt_in_channels, hgt_hidden_channels, hgt_out_channels, hgt_edge_types, nbr_plane_labels, num_layers = 3, num_heads = 8):
        super().__init__()
        # Original model had hidden_channels = 256, num_heads = 8, num_layers = 3
        self.graph_model = HGT(in_channels=hgt_in_channels,
                               hidden_channels=hgt_hidden_channels,
                               out_channels=hgt_out_channels,
                               edge_types = hgt_edge_types,
                               num_heads=num_heads,
                               num_layers=num_layers)

        # Classifier layers
        self.plane_polygon_layers = nn.Sequential(
            nn.Linear(hgt_out_channels, hgt_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hgt_out_channels, nbr_plane_labels),
            nn.ReLU(inplace=True)
        )

        self.plane_parameter_layers = nn.Sequential(
            nn.Linear(hgt_out_channels, hgt_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hgt_out_channels, 4),
        )
    def forward(self, data):
        node_features = self.graph_model(data.x_dict, data.edge_index_dict)

        # Logits for label
        plane_polygon_logits = self.plane_polygon_layers(node_features['plane'])

        # Logits for parameters
        plane_parameters = self.plane_parameter_layers(node_features['plane'])

        return plane_polygon_logits, plane_parameters



class PlaneClassifierGNN(PlaneClassifier):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Not sure if to keep this layer for something
        del self.fc1

        # Size of feature dimension in backbone output
        self.num_feats = cfg.MODEL.OUT_FEATURE_CHANNELS

        # HGT config
        self.dim_hgt_hidden = cfg.MODEL.PLANE_HEAD.HGT_DIM_HIDDEN
        self.dim_hgt_in = cfg.MODEL.PLANE_HEAD.HGT_DIM_IN
        self.hgt_classifier_connect_planes = cfg.MODEL.PLANE_HEAD.HGT_CLASSIFIER_CONNECT_PLANES
        self.minimum_average_weight_iterations = cfg.MODEL.PLANE_HEAD.HGT_CLASSIFIER_MINIMUM_AVERAGE_WEIGHT_ITERATIONS

        hgt_in_channels = {
            'junction': self.dim_hgt_in + 2,
            'line':     self.dim_hgt_in + 2,
            'plane':    self.dim_hgt_in + 2,
        }
        # Alternatives:
        # 1. Make all possible line->plane connections
        # 2. Make junction->plane connections for topK closest junctions.
        # 3. Make line->plane connections for topK closest lines.
        # 4. Use distance nodes to connect planes and lines
        hgt_edge_types=[
            ('junction', 'junction_line', 'line'),
            ('plane', 'plane_junction_hypothesis', 'junction'),
            ('plane', 'plane_line_hypothesis', 'line'),
            # If undirected
            ('line', 'rev_junction_line', 'junction'),
            ('junction', 'rev_plane_junction_hypothesis', 'plane'),
            ('line', 'rev_plane_line_hypothesis', 'plane'),
        ]
        hgt_edge_types += [(n, 'self', n) for n in hgt_in_channels.keys()]
        hgt_out_channels = self.dim_hgt_hidden
        hgt_num_layers = cfg.MODEL.PLANE_HEAD.HGT_NUM_LAYERS
        hgt_num_heads = cfg.MODEL.PLANE_HEAD.HGT_NUM_ATTN_HEADS

        self.proposal_gnn = ProposalGNN(hgt_in_channels, self.dim_hgt_hidden, hgt_out_channels, hgt_edge_types, self.nbr_labels, hgt_num_layers, hgt_num_heads)

        #TODO: Not proper, should not depend on dataset config.
        input_size = cfg.DATASETS.TARGET.WIDTH
        assert input_size == cfg.DATASETS.TARGET.HEIGHT
        self.plane_centroid_feature_conv = MultiScalePlaneConv(self.num_feats, self.dim_hgt_in, input_size, cfg.MODEL.PLANE_HEAD.HGT_ANCHORS_PER_SIDE)

        self.line_feature_conv = nn.Sequential(
            nn.Conv2d(256, self.dim_loi, 1),
            nn.ReLU(inplace=True)
            )
        self.junction_feature_conv = nn.Sequential(
            nn.Conv2d(256, self.dim_hgt_in, 3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.plane_polygon_feature_conv = nn.Sequential(
            nn.Conv2d(256, self.dim_loi, 1),
            nn.ReLU(inplace=True)
            )

        # Line pooling operation from the line classifier.
        # Used to sample lines contained in the plane sampling.
        # Since lines are sampled during training it is not possible to re-use pooled features from the line classifier.
        self.line_sampling_pooling = LinePooling(self.n_pts0, self.n_pts1, self.dim_loi)
        # FC Layers for line features
        self.fc_line2hgt = nn.Sequential(
            nn.Linear(self.n_pts1 * self.dim_loi, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_hgt_in),
            nn.ReLU(inplace=True),
            )



        if getattr(cfg.MODEL.PLANE_HEAD, 'CENTROID_LOSS_WEIGHTS', None):
            pc_label_weights = torch.tensor(cfg.MODEL.PLANE_HEAD.CENTROID_LOSS_WEIGHTS, dtype=torch.float32)
        else:
            pc_label_weights = None
        self.pc_label_loss = nn.CrossEntropyLoss(weight = pc_label_weights)

        self.hgt_topk_junction_connections = cfg.MODEL.PLANE_HEAD.HGT_TOPK_JUNCTION_CONNECTIONS
        self.hgt_topk_line_connections = cfg.MODEL.PLANE_HEAD.HGT_TOPK_LINE_CONNECTIONS
        self.hgt_polygon_line_score_threshold = cfg.MODEL.PLANE_HEAD.HGT_POLYGON_LINE_SCORE_THRESHOLD

        self.plane_line_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.MODEL.PLANE_HEAD.HGT_LINK_PREDICTION_POS_WEIGHT))

        #==================== DEBUG ---------------------------------
        # from parsing.utils.visualization import ImagePlotter
        # from parsing.utils.labels import LabelMapper
        # self.lm = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        # self.plane_labels = cfg.MODEL.PLANE_LABELS
        # self.img_viz = ImagePlotter(self.lm.get_line_labels(), self.lm.get_junction_labels(), cfg.MODEL.PLANE_LABELS)
        # self.img_counter = 0
        #==================== DEBUG ---------------------------------

        # Setup related to final classifer GNN
        hgt_in_channels['plane'] = 4*self.dim_loi+2
        hgt_edge_types=[
            ('junction', 'junction_line', 'line'),
            ('plane', 'plane_junction_hypothesis', 'junction'),
            # ('plane', 'plane_line_hypothesis', 'line'),
            # If undirected
            ('line', 'rev_junction_line', 'junction'),
            ('junction', 'rev_plane_junction_hypothesis', 'plane'),
            # ('line', 'rev_plane_line_hypothesis', 'plane'),
            ('plane', 'plane_plane_channel', 'plane'),
        ]
        hgt_edge_types += [(n, 'self', n) for n in hgt_in_channels.keys()]
        self.final_gnn = FinalGNN(hgt_in_channels, self.dim_hgt_hidden, hgt_out_channels, hgt_edge_types, self.nbr_labels, hgt_num_layers, hgt_num_heads)
        self.plane_parameter_loss = nn.SmoothL1Loss()
        self.plane_parameters_mean = torch.tensor(cfg.DATASETS.PLANE_PARAMETERS.MEAN)
        self.plane_parameters_std = torch.tensor(cfg.DATASETS.PLANE_PARAMETERS.STD)



    @classmethod
    def initialize_loss(cls, device):
        loss_dict = {
            'loss_gnn_pc_label':  torch.zeros(1, device=device),
            'loss_gnn_pc_off':  torch.zeros(1, device=device),
            'loss_gnn_plane_line_edge': torch.zeros(1, device=device),
            'loss_gnn_plane': torch.zeros(1, device=device),
            'loss_gnn_plane_iou': torch.zeros(1, device=device),
            'loss_gnn_plane_parameters': torch.zeros(1, device=device)
            }
        return loss_dict

    def initialize(self, device):
        self.loss_dict = self.initialize_loss(device)
        self.plane_parameters_mean = self.plane_parameters_mean.to(device)
        self.plane_parameters_std = self.plane_parameters_std.to(device)
        self.extra_info = {
            'time_plane_generation': 0.0,
            'time_plane_generation_final_gnn': 0.0,
            'time_plane_generation_setup_graph': 0.0,
            'time_plane_generation_get_polygons': 0.0,
            'time_plane_generation_proposal_gnn': 0.0,
            'time_plane_generation_sample_features': 0.0,
            'time_plane_read_ann': 0.0,
            'time_plane_sample_logits': 0.0,
            'time_plane_calculate_loss': 0.0,
        }
        self.loss_initialized = True

    def reset_and_get_loss_dict(self):
        self.loss_initialized = False

        # Free memory
        del self.plane_centroid_features
        del self.plane_polygon_features
        del self.line_features
        del self.junction_features

        return self.loss_dict, self.extra_info

    def _denormalize_plane_parameters(self, parameters):
        return parameters*self.plane_parameters_std + self.plane_parameters_mean

    def _get_plane_polygon_graph_features(self, sampled_polygon_features, polygon_centroids):
        normalized_coords = polygon_centroids/128.0
        # Stack with coordinates
        new_features = torch.cat([normalized_coords, sampled_polygon_features], dim=1)
        return new_features

    def _get_plane_centroid_graph_features(self, features):
        X, Y = self.plane_centroid_feature_conv.get_grid_positions_normalized()
        X_flat = X.flatten().to(features.device)
        Y_flat = Y.flatten().to(features.device)
        # Stack with coordinates
        new_features = torch.cat([X_flat[:,None], Y_flat[:,None], features.flatten(start_dim=1).T], dim=1)

        return new_features

    def _get_junction_graph_features(self, features, juncs_pred):
        h,w = features.size(1), features.size(2)
        normalized_coords = juncs_pred/128.0
        xint = juncs_pred[:,0].long().clamp(min=0, max=w-1)
        yint = juncs_pred[:,1].long().clamp(min=0, max=h-1)
        j_features = features[:,yint, xint]
        # Stack with coordinates
        new_features = torch.cat([normalized_coords, j_features.T], dim=1)

        return new_features

    def _get_line_graph_features(self, features, lines_pred):
        U,V = lines_pred[:,:2], lines_pred[:,2:]
        # Use mid point as coordinate, should we have angle as well?
        normalized_coords = (U+V)/2.0/128.0
        # Sample along line
        line_features = self.line_sampling_pooling(features, lines_pred)
        # Compress features
        line_features = self.fc_line2hgt(line_features)
        # Stack with coordinates
        new_features = torch.cat([normalized_coords, line_features], dim=1)

        return new_features

    def _plane_ann_to_poly(self, ann):
        junctions = ann['junctions']
        gt_planes_pos = [junctions[p['junction_idx']] for p in ann['planes']]
        gt_labels = torch.tensor([p['semantic'] for p in ann['planes']], dtype=torch.long, device=junctions.device)
        gt_plane_centroids = torch.stack([p['centroid'] for p in ann['planes']])

        return gt_planes_pos, gt_labels, gt_plane_centroids

    def _inference(self, plane_polygon_features, plane_centroid_features, line_features, junction_features, juncs_pred, line_output, img=None):
        ctimer = CudaTimer(self.enable_timing)
        ctimer_part = CudaTimer(self.enable_timing)
        device = plane_centroid_features.device

        ctimer.start_timer()
        ctimer_part.start_timer()

        # Setup graph input data
        data = HeteroData()

        # Node features
        data['junction'].x  = self._get_junction_graph_features(junction_features, juncs_pred)
        data['line'].x      = self._get_line_graph_features(line_features, line_output['lines_pred'])
        data['plane'].x     = self._get_plane_centroid_graph_features(plane_centroid_features)
        nbr_j = data['junction'].x.size(0)
        nbr_l = data['line'].x.size(0)
        nbr_p = data['plane'].x.size(0)

        if self.hgt_topk_junction_connections > 0:
            # Calculate nearest k junctions for each plane, set up edges
            p2j_dist2 = torch.sum((data['plane'].x[:,None,:2] - data['junction'].x[:,:2])**2, dim=-1)
            k = min(self.hgt_topk_junction_connections, nbr_j)
            p_idx = torch.arange(nbr_p).repeat_interleave(k).to(device)
            _, j_idx = p2j_dist2.topk(k, dim = 1, largest=False)
            pj_edges = torch.stack((p_idx, j_idx.flatten()))
            data['plane', 'plane_junction_hypothesis', 'junction'].edge_index = pj_edges
        elif self.hgt_topk_junction_connections < 0:
            # Connect to all junctions
            p_idx = torch.arange(nbr_p).repeat_interleave(nbr_j).to(device)
            j_idx = torch.arange(nbr_j).repeat(nbr_p).to(device)
            pj_edges = torch.stack((p_idx, j_idx))
            data['plane', 'plane_junction_hypothesis', 'junction'].edge_index = pj_edges
        # If  self.hgt_topk_junction_connections == 0 we form no connections


        if self.hgt_topk_line_connections > 0:
            # Calculate nearest k lines for each plane, set up edges
            p2l_dist2 = torch.sum((data['plane'].x[:,None,:2] - data['line'].x[:,:2])**2, dim=-1)
            k = min(self.hgt_topk_line_connections, nbr_l)
            p_idx = torch.arange(nbr_p).repeat_interleave(k).to(device)
            _, l_idx = p2l_dist2.topk(k, dim = 1, largest=False)
            pl_edges = torch.stack((p_idx, l_idx.flatten()))
            data['plane', 'plane_line_hypothesis', 'line'].edge_index = pl_edges
        elif self.hgt_topk_line_connections < 0:
            # Connect to all lines
            p_idx = torch.arange(nbr_p).repeat_interleave(nbr_l).to(device)
            l_idx = torch.arange(nbr_l).repeat(nbr_p).to(device)
            pl_edges = torch.stack((p_idx, l_idx))
            data['plane', 'plane_line_hypothesis', 'line'].edge_index = pl_edges
        # If  self.hgt_topk_line_connections == 0 we form no connections


        # Calculate edges for junctions and lines
        l_idx = torch.arange(data['line'].x.size(0)).repeat_interleave(2).to(device)
        j_idx = line_output['edges_pred'].flatten()
        jl_edges = torch.stack((j_idx, l_idx))
        data['junction', 'junction_line', 'line'].edge_index = jl_edges

        # Add self edges
        for node_type in data.metadata()[0]:
            sequence = torch.arange(data[node_type].x.size(0), device = data[node_type].x.device)
            data[node_type, 'self', node_type].edge_index = torch.stack([sequence,sequence])

        # Add reverse directions
        data = T.ToUndirected()(data)

        self.extra_info['time_plane_generation_setup_graph'] += ctimer_part.end_timer()
        ctimer_part.start_timer()

        # Run through proposal GNN
        pc_label_logits, pc_offset_logits, plane_line_logits = self.proposal_gnn(data)
        self.extra_info['time_plane_generation_proposal_gnn'] += ctimer_part.end_timer()

        # Generate polygons
        ctimer_part.start_timer()
        hyp_planes_node_idx, success_mask = self._get_polygons(line_output['edges_pred'],
                                                               plane_line_logits,
                                                               device=device,
                                                               juncs_pred=juncs_pred,
                                                               img = img)
        self.extra_info['time_plane_generation_get_polygons'] += ctimer_part.end_timer()

        # Sample polygon features

        hyp_planes_pos = [juncs_pred[p] for p in hyp_planes_node_idx]
        classifier_output = {}
        if hyp_planes_pos:
            ctimer_part.start_timer()
            sampled_polygon_features, plane_masks, polygon_centroids = self._sample_polygon_feature(plane_polygon_features, hyp_planes_pos)
            self.extra_info['time_plane_generation_sample_features'] += ctimer_part.end_timer()

            # Run in final HGN network
            ctimer_part.start_timer()
            data_final = data.node_type_subgraph({'junction', 'line'})
            del data # Free memory
            data_final['plane'].x = self._get_plane_polygon_graph_features(sampled_polygon_features, polygon_centroids)
            nbr_p = polygon_centroids.size(0)
            p_idx = torch.repeat_interleave(torch.arange(nbr_p), torch.tensor([p.numel() for p in hyp_planes_node_idx])).to(device)
            j_idx = torch.cat(hyp_planes_node_idx, dim=0).to(device)
            data_final['plane', 'plane_junction_hypothesis', 'junction'].edge_index = torch.stack([p_idx, j_idx])
            data_final['junction', 'rev_plane_junction_hypothesis', 'plane'].edge_index = torch.stack([j_idx, p_idx])
            data_final['plane', 'self', 'plane'].edge_index = torch.stack([p_idx, p_idx])
            if self.hgt_classifier_connect_planes:
                p_idx = torch.arange(nbr_p).repeat_interleave(nbr_p).to(device)
                p_idx2 = torch.arange(nbr_p).repeat(nbr_p).to(device)
                keep_edges = (p_idx2 != p_idx)
                pp_edges = torch.stack((p_idx[keep_edges], p_idx2[keep_edges]))
                data_final['plane', 'plane_plane_channel', 'plane'].edge_index = pp_edges

            classifier_output['plane_polygon_logits'], classifier_output['plane_parameters'] = self.final_gnn(data_final)
            classifier_output['polygon_centroids'] = polygon_centroids
            classifier_output['polygon_masks'] = plane_masks
            self.extra_info['time_plane_generation_final_gnn'] += ctimer_part.end_timer()

        self.extra_info['time_plane_generation'] += ctimer.end_timer()
        output = {
            'pc_label_logits':      pc_label_logits,
            'pc_offset_logits':     pc_offset_logits,
            'plane_line_logits':    plane_line_logits,
            'hyp_planes_node_idx':  hyp_planes_node_idx,
            'hyp_planes_pos':       hyp_planes_pos,
            'success_mask':       success_mask,
        }
        output.update(classifier_output)

        return output


    #TODO: Done on CPU, probably one of the bottlenecks.
    def _find_minimum_average_weight_cycle(self, edges_pred, pscores, juncs_pred):
        if edges_pred.size(0) < 3:
            return False, None

        iterations = min(self.minimum_average_weight_iterations,edges_pred.size(0))

        # Initialize graph
        inv_pscores = 1-pscores
        edge_with_weight = [[int(e[0]), int(e[1]), float(s)] for e,s in zip(edges_pred, inv_pscores)]
        nx_graph_full = nx.Graph()
        nx_graph_full.add_weighted_edges_from(edge_with_weight)

        best_average_cost = 100
        best_polygon = None

        # Remove the highest scoring edge
        _, topk_indices = torch.topk(pscores, iterations, sorted=False)
        for max_edge_idx in topk_indices:
            nx_graph = nx_graph_full.copy()
            starting_edge = edges_pred[max_edge_idx].tolist()
            nx_graph.remove_edge(*starting_edge)
            try:
                polygon = nx.shortest_path(nx_graph, source = starting_edge[0], target=starting_edge[1], weight='weight')
                # Make cycle
                polygon.append(polygon[0])
                average_cost = np.mean([nx_graph_full[e0][e1]['weight'] for e0, e1 in zip(polygon[:-1],polygon[1:])])
                sg_polygon = sg.Polygon(juncs_pred[np.array(polygon)])
                if average_cost < best_average_cost and sg_polygon.is_valid:# and sg_polygon.area > 100:
                    best_polygon = polygon
                    best_average_cost = average_cost

            except nx.NetworkXNoPath:
                pass

        if best_polygon:
            return True, best_polygon
        else:
            return False, None




    def _get_polygons(self, edges_pred, plane_line_logits, device, juncs_pred, img = None, centroids = None):

        # Construct a networkx graph with junctions and lines
        plane_line_score = plane_line_logits.sigmoid()
        plane_polygons = []
        plane_polygons_set = set()
        success_mask = torch.zeros(plane_line_logits.size(0), dtype=torch.bool)
        juncs_pred_np = juncs_pred.to("cpu").numpy()
        for p_idx, pscores in enumerate(plane_line_score):
            p_mask = pscores>self.hgt_polygon_line_score_threshold
            if not p_mask.any():
                success_mask[p_idx] = False
                continue
            edges_pred_reduced = edges_pred[p_mask].to('cpu')
            pscores = pscores[p_mask].to('cpu')
            #DEBUG
            # import sys
            # print(img.shape)
            # fig, ax = self.img_viz.no_border_imshow(img, autoscale=True)
            # lines = juncs_pred[edges_pred_reduced].to('cpu').numpy()*512/128
            # X, Y = self.plane_centroid_feature_conv.get_grid_positions()
            # centroids = torch.stack([X.flatten(), Y.flatten()], dim=1)
            # ax.plot([lines[:,0,0],lines[:,1,0]],
            #         [lines[:,0,1],lines[:,1,1]],
            #         color = 'b',
            #         linewidth = self.img_viz.LINE_WIDTH)
            # ax.plot(*centroids[p_idx].to('cpu').numpy()*512/128,
            #         color = 'g',
            #         marker='s')
            # for lscore, lpos in zip(pscores, lines):
            #     text_pos = np.clip((lpos[0]+lpos[1])/2, 30, 480)
            #     ax.text(*text_pos, f'S{lscore:0.2f}', color='white', backgroundcolor='black')

            #DEBUG END
            success, polygon = self._find_minimum_average_weight_cycle(edges_pred_reduced, pscores, juncs_pred_np)
            success_mask[p_idx] = success
            if success:
                polygon_set_entry = frozenset(polygon)
                if polygon_set_entry not in plane_polygons_set:
                    plane_polygons.append(torch.tensor(polygon, device=device, dtype=torch.long))
                    plane_polygons_set.add(polygon_set_entry)
                # ax.plot(*juncs_pred[torch.tensor(polygon)].to('cpu').numpy().T*512/128,
                #         color = 'g',
                #         linewidth = self.img_viz.LINE_WIDTH)
                # ax.text(*centroids[p_idx].to('cpu').numpy()*512/128, f'AS{average_score:0.2f}', color='black', backgroundcolor='white')

        #     plt.savefig(f'../../debug/line_plane_edges/scores_{self.img_counter}_{p_idx}.png')
        #     plt.close()
        # self.img_counter +=1

        return plane_polygons, success_mask


    def forward_test(self, juncs_pred, line_output, features, img=None):

        device = juncs_pred.device
        self.initialize(device=device)
        if line_output['lines_pred'].numel() == 0:
            return {'planes_pred': []}, self.extra_info

        ctimer = CudaTimer(self.enable_timing)

        line_output = self._filter_topk_and_score(line_output, keys = ['edges_pred','lines_pred', 'lines_valid_score'])

        plane_centroid_features = self.plane_centroid_feature_conv(features)[0]
        plane_polygon_features = self.plane_polygon_feature_conv(features)[0]
        line_features = self.line_feature_conv(features)[0]
        junction_features = self.junction_feature_conv(features)[0]
        inf_output = self._inference(
            plane_polygon_features,
            plane_centroid_features,
            line_features,
            junction_features,
            juncs_pred,
            line_output,
            img = img
            )


        X, Y = self.plane_centroid_feature_conv.get_grid_positions()
        pcentroid_pred = torch.stack((X.flatten(), Y.flatten()), dim=1).to(device)
        pcentroid_pred += scaled_sigmoid(inf_output['pc_offset_logits'], offset=-0.5, scale=float(self.plane_centroid_feature_conv.stride))


        pcentroid_score = inf_output['pc_label_logits'].softmax(1)
        pcentroid_label = pcentroid_score[:,1:].argmax(dim=1) + 1
        # pcentroid_label = pcentroid_score.argmax(dim=1)
        pcentroid_valid_score = 1-pcentroid_score[:,0]
        pcentroid_label[pcentroid_valid_score < 0.05] = 0
        pcentroid_label_score = torch.gather(pcentroid_score, 1, pcentroid_label.unsqueeze(1)).squeeze(1)

        output = {
            'pcentroid_pred': pcentroid_pred,
            'pcentroid_label_score': pcentroid_label_score,
            'pcentroid_valid_score': pcentroid_valid_score,
            'pcentroid_score': pcentroid_score,
            'pcentroid_label': pcentroid_label,

        }

        hyp_planes_node_idx = inf_output['hyp_planes_node_idx']

        if hyp_planes_node_idx:
            plane_score = inf_output['plane_polygon_logits'].softmax(1)
            plane_label = plane_score[:,1:].argmax(dim=1) + 1 # Never classify as background
            # plane_label = plane_score.argmax(dim=1)
            plane_valid_score = 1-plane_score[:,0]
            plane_label[plane_valid_score < 0.05] = 0
            plane_label_score = torch.gather(plane_score, 1, plane_label.unsqueeze(1)).squeeze(1)
            plane_parameters = inf_output['plane_parameters']
            if self.plane_nms_threshold > 0:
                keep_idx = self._nms_iou_simple(inf_output['polygon_masks'], plane_label_score)
                plane_score = plane_score[keep_idx]
                plane_valid_score = plane_valid_score[keep_idx]
                plane_label_score = plane_label_score[keep_idx]
                plane_label = plane_label[keep_idx]
                plane_parameters = plane_parameters[keep_idx]
                hyp_planes_node_idx = [hyp_planes_node_idx[i] for i in keep_idx]

            output_plane = {
                'planes_score': plane_score,
                'planes_label': plane_label,
                'planes_label_score': plane_valid_score,
                'planes_valid_score': plane_label_score,
                'planes_pred': hyp_planes_node_idx,
                'planes_parameters': self._denormalize_plane_parameters(plane_parameters)
            }
        else:
            output_plane = {
                'planes_pred': hyp_planes_node_idx
            }
        output.update(output_plane)


        return output, self.extra_info

    def forward_train_batch(self, features):
        self.plane_centroid_features = self.plane_centroid_feature_conv(features)
        self.line_features = self.line_feature_conv(features)
        self.junction_features = self.junction_feature_conv(features)
        self.plane_polygon_features = self.plane_polygon_feature_conv(features)

    def _filter_topk_and_score(self, line_output, keys = ['edges_pred','lines_pred']):
        if 'lines_valid_score' in line_output:
            valid_score = line_output['lines_valid_score']
        else:
            scores = line_output['lines_logits'].softmax(1)
            valid_score = 1-scores[:,0]

        score_mask = valid_score > self.edge_valid_score_threshold
        line_output = {
            k: line_output[k][score_mask] for k in keys
        }
        valid_score = valid_score[score_mask]

        if self.topk_edges < line_output['lines_pred'].size(0):
            _, topk_indices = torch.topk(valid_score, self.topk_edges, sorted=False)
            line_output = {
                k: line_output[k][topk_indices] for k in keys
            }
        return line_output


    def forward_train(self, img_idx, juncs_pred, line_output, meta, ann, batch_size, img=None, batch_idx = None):
        device = juncs_pred.device
        ctimer = CudaTimer(self.enable_timing)

        if not self.loss_initialized:
            self.initialize(device)

        line_output = self._filter_topk_and_score(line_output, keys = ['edges_pred','lines_pred'])

        # Get annotated planes
        ctimer.start_timer()
        gt_planes_pos, gt_labels, gt_plane_centroids = self._plane_ann_to_poly(ann)
        self.extra_info['time_plane_read_ann'] += ctimer.end_timer()

        inf_output = self._inference(
            self.plane_polygon_features[img_idx],
            self.plane_centroid_features[img_idx],
            self.line_features[img_idx],
            self.junction_features[img_idx],
            juncs_pred,
            line_output,
            img = img
            )

        # Do piparte matching and construct loss
        ctimer.start_timer()
        X, Y = self.plane_centroid_feature_conv.get_grid_positions()
        centroid_sample_coords = torch.stack((X.flatten(), Y.flatten()), dim=1)

        cost_matrix = torch.sum((centroid_sample_coords[:,None] - gt_plane_centroids.to('cpu'))**2, dim=-1)
        plane_pred_ind, plane_gt_ind = linear_sum_assignment(cost_matrix.numpy())
        centroid_extended_gt = torch.zeros(centroid_sample_coords.size(0), device=device, dtype=torch.long)
        centroid_extended_gt[plane_pred_ind] = gt_labels[plane_gt_ind]

        self.loss_dict['loss_gnn_pc_label'] += self.pc_label_loss(inf_output['pc_label_logits'], centroid_extended_gt) / batch_size

        #Offset loss for matched centroids
        pc_offset_logits = inf_output['pc_offset_logits'][plane_pred_ind]
        target_offset = gt_plane_centroids[plane_gt_ind] - centroid_sample_coords[plane_pred_ind].to(device)

        #Make loss invariant to scale
        target_offset /= float(self.plane_centroid_feature_conv.stride)

        self.loss_dict['loss_gnn_pc_off'] += sigmoid_l1_loss(pc_offset_logits, target_offset, offset=-0.5) / batch_size

        # Calculate edge prediction loss
        # Find corresponding edges with piparte matching, contruct a loss term per GT plane
        with torch.no_grad():
            # L1 distance to make it less sensitive to outliers.
            jgt2jp_dist2 = torch.sum(torch.abs(ann['junctions'][:,None] - juncs_pred), axis=-1)
            lgt2lp_dist2 = jgt2jp_dist2[ann['edges_positive'][:,None], line_output['edges_pred']].sum(axis=-1)
            lgt2lp_dist2_rev = jgt2jp_dist2[ann['edges_positive'].roll(1,1)[:,None], line_output['edges_pred']].sum(axis=-1)
            lgt2lp_dist2 = torch.minimum(lgt2lp_dist2, lgt2lp_dist2_rev)
            line_gt_ind, line_pred_ind = linear_sum_assignment(lgt2lp_dist2.to('cpu').numpy())
            include_mask = lgt2lp_dist2[line_gt_ind,line_pred_ind].to('cpu').numpy() < 15
            line_gt_ind = line_gt_ind[include_mask]
            line_pred_ind = line_pred_ind[include_mask]

        for p_pred_idx, p_gt_idx in zip(plane_pred_ind, plane_gt_ind):
            # Find GT edges that should belong to plane
            plane_line_gt = torch.zeros(ann['edges_positive'].size(0), device=device, dtype=torch.float32)
            plane_line_gt[ann['planes'][p_gt_idx]['edge_idx']] = 1

            # Match GT edges according to the matching for predicted lines and gt lines
            line_extended_gt = torch.zeros(line_output['lines_pred'].size(0), device=device, dtype=torch.float32)
            line_extended_gt[line_pred_ind] = plane_line_gt[line_gt_ind]
            #-------------------DEBUG--------------------
            # with torch.no_grad():
            #     fig, ax = plt.subplots(2,1)
            #     plot_gt_lines = ann['junctions'][ann['edges_positive']]
            #     plot_gt_lines = plot_gt_lines[plane_line_gt>0].to('cpu').numpy()*512/128
            #     plot_pred_lines = line_output['lines_pred'][line_extended_gt>0].to('cpu').numpy()*512/128
            #     self.img_viz.no_border_imshow(img, ax=ax[0], autoscale=False)
            #     ax[0].plot([plot_gt_lines[:,0,0],plot_gt_lines[:,1,0]],
            #                [plot_gt_lines[:,0,1],plot_gt_lines[:,1,1]],
            #               color = 'b',
            #               linewidth = self.img_viz.LINE_WIDTH)
            #     ax[0].set_title('GT')
            #     self.img_viz.no_border_imshow(img, ax=ax[1], autoscale=False)
            #     ax[1].plot([plot_pred_lines[:,0],plot_pred_lines[:,2]],
            #                [plot_pred_lines[:,1],plot_pred_lines[:,3]],
            #               color = 'b',
            #               linewidth = self.img_viz.LINE_WIDTH)
            #     ax[1].set_title('Pred')
            #
            #     fname = ann['filename']
            #     plt.savefig(f'/host_home/debug/line_plane_edges_train/{fname}_{p_pred_idx}.png')
            #     plt.close()

            #-------------------DEBUG--------------------

            self.loss_dict['loss_gnn_plane_line_edge'] += self.plane_line_bce_loss(inf_output['plane_line_logits'][p_pred_idx], line_extended_gt) / batch_size / centroid_sample_coords.size(0)

        # Loss for polygon
        if inf_output['hyp_planes_node_idx']:
            p_ind, gt_ind, iou, loss = self.eval.evaluate_planes_biparte(
                gt_planes_pos,
                gt_labels,
                inf_output['hyp_planes_pos'],
                inf_output['plane_polygon_logits'],
                metrics= ['loss', 'iou'],
                pred_masks = inf_output['polygon_masks'])

            self.loss_dict['loss_gnn_plane'] += loss['label'] / batch_size
            self.loss_dict['loss_gnn_plane_iou'] += loss['iou'] / batch_size

            # Loss for parameters
            if len(gt_ind) > 0:
                parameter_gt = torch.stack([ann['planes'][idx]['parameters'] for idx in gt_ind])
                self.loss_dict['loss_gnn_plane_parameters'] += self.plane_parameter_loss(inf_output['plane_parameters'][p_ind], parameter_gt) / batch_size

            #-----------------------# DEBUG----------------------
            # from parsing.utils.comm import to_device
            # from math import ceil
            # from matplotlib.patches import ConnectionPatch
            # import sys
            # nbr_proposals = inf_output['polygon_centroids'].size(0)
            # nbr_matched_gt = len(plane_gt_ind)
            # nbr_rows = 5
            # nbr_cols = ceil((nbr_proposals+nbr_matched_gt)/5)
            # fig, axes = plt.subplots(nbr_rows,nbr_cols,sharex = True, sharey = True, dpi = 100, figsize=(16,9))
            # ax_idx = 0
            # img_np = np.rollaxis(img.to('cpu').numpy(), 0, 3)
            # plane_score = inf_output['plane_polygon_logits'].softmax(1)
            # plane_label = plane_score.argmax(dim=1)
            # plane_valid_score = 1-plane_score[:,0]
            # plane_label_score = torch.gather(plane_score, 1, plane_label.unsqueeze(1)).squeeze(1)
            # for j_idx, p_score in zip(inf_output['hyp_planes_node_idx'], plane_label_score):
            #     ax = axes.flat[ax_idx]
            #     ax.imshow(img_np)
            #     ax.axis('off')
            #
            #     poly = juncs_pred[j_idx].to('cpu').numpy()*512/128
            #     self.img_viz._add_polygon_patch(ax, poly, 'b', skip_hatch = True)
            #     ax.set_title(f'P{ax_idx}_S{p_score:0.2f}')
            #     ax_idx += 1
            #
            # # Plot gt
            # for p_idx, gt_idx in zip(plane_pred_ind, plane_gt_ind):
            #     ax = axes.flat[ax_idx]
            #     ax.imshow(img_np)
            #     ax.axis('off')
            #
            #     poly = gt_planes_pos[gt_idx].to('cpu').numpy()*512/128
            #     self.img_viz._add_polygon_patch(ax, poly, 'g', skip_hatch = True)
            #     ax.set_title(f'GT{gt_idx}->P{p_idx}')
            #     con = ConnectionPatch(xyA=(256,256), xyB=(256,256), coordsA="data", coordsB="data",
            #           axesA=ax, axesB=axes.flat[p_idx], color="r")
            #     ax.add_artist(con)
            #     ax_idx += 1
            #
            # plt.savefig(f'/proj/wireframe-layout-est/users/x_dagil/debug/plane_proposals/b{batch_idx}_i{img_idx}.png')
            # plt.close()

            #-----------------------# END DEBUG----------------------





        self.extra_info['time_plane_calculate_loss'] += ctimer.end_timer()


    def _setup_sampling(self, cfg):
        # So that we can use PlaneClassifier as parent class
        pass
