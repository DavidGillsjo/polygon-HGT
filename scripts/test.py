import torch
import parsing

from parsing.utils.comm import to_device
from parsing.dataset import build_test_dataset,build_gnn_test_dataset, move_datasets
from parsing.detector import WireframeDetector
from parsing.plane_from_gt_detector import PlaneFromGtDetector
from parsing.modules.gnn import WireframeGNNClassifier
from parsing.utils.logger import setup_logger, wandb_init
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.utils.visualization import ImagePlotter
from parsing.utils.labels import LabelMapper
from torch.utils.tensorboard import SummaryWriter
import parsing.utils.metric_evaluation as me
from parsing.utils.logger import CudaTimer
import os
import os.path as osp
from skimage import io
import time
import argparse
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
import json
from glob import glob
import numpy as np
from tabulate import tabulate
from parsing.dataset.transforms import Compose, ResizeImage, ToTensor, Normalize
import networkx as nx
import wandb
from datetime import datetime
import queue

#Multi gpu
import torch.multiprocessing as mp
import multiprocessing as py_mp
import parsing.utils.multi_gpu as multi_gpu

POISON_PILL = 1234


# Handles a single process to test and evaluate
# TODO: Evaluate model in hogwild style, then return to caller and process result and plots in background.
class ModelTester:
    # Static queue to gather results to main process during multi GPU testing/training

    @staticmethod
    def setup_queue():
        m = py_mp.Manager()
        return m.Queue(maxsize=10)

    def __init__(self, cfg, validation = False, output_dir = None, nbr_plots = 5, write_tb = True, write_latex = False, show_legend = True, wandb_run = None, disable_wandb = False, result_queue = None, resume_wandb = False):
        self.cfg = cfg
        self.validation = validation
        self.output_dir = output_dir if output_dir else cfg.OUTPUT_DIR
        self.plot_dir = osp.join(self.output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        self.nbr_plots = nbr_plots
        self.logger = multi_gpu.get_logger()
        write_tb &= multi_gpu.is_main_proc()
        self.tb_logger = SummaryWriter(output_dir) if write_tb else None
        self.wandb_run = wandb_run
        self.show_legend = show_legend
        self.disable_wandb = disable_wandb
        self.resume_wandb = resume_wandb

        self.img_transform =  Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                         cfg.DATASETS.IMAGE.WIDTH),
             ToTensor(),
             Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                       cfg.DATASETS.IMAGE.PIXEL_STD,
                       cfg.DATASETS.IMAGE.TO_255)
            ]
        )
        self.score_threshold = cfg.SCORE_THRESHOLD
        self.line_nms_threshold2 = max(0,cfg.LINE_NMS_THRESHOLD)**2
        self.graph_nms = cfg.GRAPH_NMS

        write_latex &= multi_gpu.is_main_proc()
        self.write_latex = write_latex
        if write_latex:
            self.latex_dir = osp.join(self.output_dir, 'latex')
            os.makedirs(self.latex_dir, exist_ok=True)


        self.lm = LabelMapper(cfg.MODEL.LINE_LABELS, cfg.MODEL.JUNCTION_LABELS, disable=cfg.DATASETS.DISABLE_CLASSES)
        self.plane_labels = cfg.MODEL.PLANE_LABELS
        self.img_viz = ImagePlotter(self.lm.get_line_labels(), self.lm.get_junction_labels(), cfg.MODEL.PLANE_LABELS)
        self.nbr_poision_pills = 0
        self.result_queue = result_queue
        self.device = multi_gpu.get_device(cfg)

        # Skip sAP if GT plane classifier
        self.calculate_sap = 'GT_PLANE_CLASSIFIER' != self.cfg.MODEL.NAME.upper()
        # Calculate Plance Centroid sAP if GT plane classifier
        self.calculate_pcap = 'GT_PLANE_CLASSIFIER' == self.cfg.MODEL.NAME.upper()

    
    @property
    def datasets(self):
        if 'GNN' in self.cfg.MODEL.NAME.upper():
            return build_gnn_test_dataset(self.cfg, validation = self.validation)
        else:
            return build_test_dataset(self.cfg, validation = self.validation)
    
    def _init_model(self):
        if 'GNN' in self.cfg.MODEL.NAME.upper():
            model = WireframeGNNClassifier(self.cfg)
        elif 'GT_PLANE_CLASSIFIER' == self.cfg.MODEL.NAME.upper():
            model = PlaneFromGtDetector(self.cfg)
        elif 'HOURGLASS' == self.cfg.MODEL.NAME.upper():
            model = WireframeDetector(self.cfg)
        else:
            raise NotImplementedError(f'Model {cfg.MODEL.NAME} not supported')

        model = model.to(self.device)
        checkpointer = DetectronCheckpointer(self.cfg,
                                             model,
                                             save_dir=self.cfg.OUTPUT_DIR,
                                             save_to_disk=True,
                                             logger=self.logger)
        return model, checkpointer

    def _init_wandb(self, cfg, checkpointer):
        # Do not create a new one if existing
        if self.wandb_run or not multi_gpu.is_main_proc():
            return
        timestamp = datetime.now().isoformat(timespec='seconds')
        self.wandb_run = wandb_init(cfg, checkpointer, resume=self.resume_wandb, timestamp = timestamp, disable_wandb = self.disable_wandb)

    def _load_model_from_cfg(self):
        model, checkpointer= self._init_model()

        if getattr(self.cfg, 'CHECKPOINT'):
            checkpointer.load(f=self.cfg.CHECKPOINT, use_latest=False)
        else:
            checkpointer.load()

        self._init_wandb(self.cfg, checkpointer)

        # For multi gpu
        model = multi_gpu.wrap_model(model)

        return model


    def _get_all_models(self, subsample = 1):
        model, checkpointer= self._init_model()

        checkpoint_path = getattr(self.cfg, 'CHECKPOINT', None)
        if checkpoint_path:
            model_paths = glob(osp.join(checkpoint_path, '*.pth'))
        else:
            model_paths = glob(osp.join(self.cfg.OUTPUT_DIR, '*.pth'))

        for model_path in sorted(model_paths):
            model_name = osp.basename(model_path)
            epoch = int(model_name[6:11])
            if (epoch % subsample) == 0:
                checkpointer.load(f=model_path, use_latest = False)
                self._init_wandb(self.cfg, checkpointer)
                # For multi gpu
                model = multi_gpu.wrap_model(model)
                yield (epoch, model)


    def _log_fig(self, fig, fname, epoch):
        if self.tb_logger is not None:
            self.tb_logger.add_figure(fname, fig, global_step=epoch, close=True)
        if self.wandb_run is not None:
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.wandb_run.log({
                fname: wandb.Image(image_from_plot),
                'epoch': epoch
                })

    def _tb_plots(self, epoch, im, im_name, output, extra_info, idx = None):
        filetype ='pdf'
        # Plot predictions with higher score than threshold
        fig = self.img_viz.plot_pred(im, im_name, output, extra_info, score_threshold = self.score_threshold, ignore_invalid_junc = True, show_legend=self.show_legend)
        fname = '{}_E{:02}.{}'.format(osp.splitext(im_name)[0], epoch,filetype) if idx is None else 'E{:02}I{:02}_lj.{}'.format(epoch, idx,filetype)
        fig_path = osp.join(self.plot_dir, fname)
        plt.savefig(fig_path)

        fname = '{}_lines_junctions'.format(osp.splitext(im_name)[0]) if idx is None else 'I{:02}_lines_junctions'.format(idx)
        self._log_fig(fig, fname, epoch)
        plt.close(fig)

        # Plot top scoring planes
        fig = self.img_viz.plot_pred_top_planes(im, im_name, output, show_legend=self.show_legend)
        fname = '{}_E{:02}_top_planes.{}'.format(osp.splitext(im_name)[0], epoch,filetype) if idx is None else 'E{:02}I{:02}_top_planes.{}'.format(epoch, idx,filetype)
        fig_path = osp.join(self.plot_dir, fname)
        plt.savefig(fig_path)

        fname = '{}_top_planes'.format(osp.splitext(im_name)[0]) if idx is None else 'I{:02}_top_planes'.format(idx)
        self._log_fig(fig, fname, epoch)
        plt.close(fig)

        # Plot All Line predictions
        fig = self.img_viz.plot_final_pred(im, im_name, output, score_threshold = -1, ignore_invalid_junc = True, show_legend=False, skip_planes=True)
        fname = '{}_E{:02}_all_lines.{}'.format(osp.splitext(im_name)[0], epoch,filetype) if idx is None else 'E{:02}I{:02}_all_lines.{}'.format(epoch, idx,filetype)
        fig_path = osp.join(self.plot_dir, fname)
        plt.savefig(fig_path)
        plt.close(fig)

        # Plot Only Plane predictions
        fig = self.img_viz.plot_final_pred(im, im_name, output, score_threshold = self.score_threshold, ignore_invalid_junc = True, show_legend=self.show_legend, skip_lines=True)
        fname = '{}_E{:02}_planes.{}'.format(osp.splitext(im_name)[0], epoch,filetype) if idx is None else 'E{:02}I{:02}_planes.{}'.format(epoch, idx,filetype)
        fig_path = osp.join(self.plot_dir, fname)
        plt.savefig(fig_path)
        plt.close(fig)

        # Plot Only Plane predictions no hatch
        fig = self.img_viz.plot_final_pred(im, im_name, output, score_threshold = self.score_threshold, ignore_invalid_junc = True, show_legend=self.show_legend, skip_lines=True, skip_hatch = True)
        fname = '{}_E{:02}_planes_nohatch.{}'.format(osp.splitext(im_name)[0], epoch,filetype) if idx is None else 'E{:02}I{:02}_planes_nohatch.{}'.format(epoch, idx,filetype)
        fig_path = osp.join(self.plot_dir, fname)
        plt.savefig(fig_path)
        plt.close(fig)


    def _tb_graph(self, model):
        if not self.tb_logger:
            return
        # Create graph from first batch in first dataset
        name, dataset = next(iter(self.datasets))
        images, annotations = next(iter(dataset))
        self.tb_logger.add_graph(model, to_device(images, self.device))

    def _make_AP_latex_table(self, ap_dict, ap_name = 'sAP', epoch = 0):
        if not self.write_latex:
            return
        for mtype, thresh_dict in ap_dict.items():
            print('mtype',mtype)
            thresholds = list(thresh_dict.keys())
            t_headers = []
            for t in thresholds:
                print('t',t)
                th = 'm' if hasattr(t,'strip') else '{:g}'.format(t)
                print(th)
                t_headers.append(fr'$\text{{{ap_name}}}^{{{th}}}$')
            labels = sorted(thresh_dict[thresholds[0]].keys())
            rows = []
            for l in labels:
                print(l)
                print([thresh_dict[t][l] for t in thresholds])
                rows.append([l] + ['{:.4f}'.format(thresh_dict[t][l]) for t in thresholds])
            table = tabulate(rows, headers = ['Type'] + t_headers, tablefmt="latex_raw")

            with open(osp.join(self.latex_dir, 'E{:02}_{}_{}_table.tex'.format(epoch, ap_name,mtype)), 'w') as f:
                f.write(table)

    # def _make_room_latex_table(self, thresh_dict, epoch = 0):
    #     if not self.write_latex:
    #         return
    #     thresholds = list(thresh_dict.keys())
    #     t_headers = []
    #     for t in thresholds:
    #         th = 'm' if hasattr(t,'strip') else '{:g}'.format(t)
    #         t_headers.append(fr'${t:g}$')
    #
    #     for t, metric_dict in thresh_dict.items():
    #
    #         labels = sorted(thresh_dict[thresholds[0]].keys())
    #         rows = []
    #         for l in labels:
    #             rows.append([l] + ['{:.1f}'.format(thresh_dict[t][l]) for t in thresholds])
    #         table = tabulate(rows, headers = ['Type'] + t_headers, tablefmt="latex_raw")
    #
    #         with open(osp.join(self.latex_dir, 'E{:02}_{}_{}_table.tex'.format(epoch, ap_name,mtype)), 'w') as f:
    #             f.write(table)


    def _nms_remove(self, remove_mask, output):
        if isinstance(output['lines_pred'], torch.Tensor):
            t_mask = torch.tensor(~remove_mask)
            output['lines_pred'] = output['lines_pred'][t_mask]
            output['lines_label_score'] = output['lines_label_score'][t_mask]
            output['lines_valid_score'] = output['lines_valid_score'][t_mask]
            output['lines_label'] = output['lines_label'][t_mask]
            output['lines_score'] = output['lines_score'][t_mask]
            if 'edges_pred' in output:
                output['edges_pred'] = output['edges_pred'][t_mask]
        else:
            keep_idx = np.flatnonzero(~remove_mask)
            output['lines_pred'] = [output['lines_pred'][idx] for idx in keep_idx]
            output['lines_label_score'] = [output['lines_label_score'][idx] for idx in keep_idx]
            output['lines_valid_score'] = [output['lines_valid_score'][idx] for idx in keep_idx]
            output['lines_label'] = [output['lines_label'][idx] for idx in keep_idx]
            output['lines_score'] = [output['lines_score'][idx] for idx in keep_idx]
            if 'edges_pred' in output:
                output['edges_pred'] = [output['edges_pred'][idx] for idx in keep_idx]


    def _nms_distance(self, output):
        line_segments = np.array(output['lines_pred'], dtype=np.float32)
        if line_segments.shape[0] < 2:
            return
        line_segments[:,0] *= 128/float(output['width'])
        line_segments[:,1] *= 128/float(output['height'])
        line_segments[:,2] *= 128/float(output['width'])
        line_segments[:,3] *= 128/float(output['height'])
        line_segments = line_segments.reshape(-1,2,2)[:,:,::-1]
        score = np.array(output['lines_label_score'], dtype=np.float32)
        labels = np.array(output['lines_label'], dtype=int)
        remove_mask = np.zeros(line_segments.shape[0], dtype=bool)
        for label_idx in range(1,self.lm.nbr_line_labels()):
            label_mask = labels==label_idx
            global_idx = np.flatnonzero(label_mask)
            label_ls = line_segments[label_mask]
            label_score = score[label_mask]

            diff = ((label_ls[:, None, :, None] - label_ls[:, None]) ** 2).sum(-1)
            diff = np.minimum(
                diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
            )
            diff[np.arange(label_ls.shape[0]), np.arange(label_ls.shape[0])] = np.inf
            for g_idx, l_diff in zip(global_idx, diff):
                matches_score = label_score[l_diff < self.line_nms_threshold2]
                if matches_score.size > 0:
                    remove_mask[g_idx] |= np.any(matches_score > score[g_idx])

        if np.any(remove_mask):
            self._nms_remove(remove_mask, output)

    def _nms_graph(self, output, ax=None):
        labels = np.array(output['lines_label'], dtype=np.int)
        if labels.shape[0] < 2:
            return
        line2junc_idx = np.array(output['line2junc_idx'], dtype=np.int)
        edge2idx = {frozenset(edge):idx for idx,edge in enumerate(line2junc_idx.tolist())}
        if self.lm.nbr_line_labels()>2:
            graph_line2junc_idx = line2junc_idx[labels>0]
            graph_line2junc_idx = [tuple(e) for e in graph_line2junc_idx.tolist()]
        else:
            graph_line2junc_idx = [tuple(e) for e in line2junc_idx.tolist()]

        G = nx.Graph()
        G.add_nodes_from(range(len(output['juncs_label'])))
        G.add_edges_from(graph_line2junc_idx)
        # G = nx.OrderedGraph(graph_line2junc_idx)
        if 'proper' in self.lm.get_junction_labels():
            junction_scores = np.array(output['juncs_score'])
            j_labels = np.argmax(junction_scores[:,1:], axis=1) + 1
        else:
            j_labels = np.array(output['juncs_label'], dtype=np.int)

        score = np.array(output['lines_label_score'], dtype=np.float32)
        remove_mask = np.zeros(line2junc_idx.shape[0], dtype=np.bool)

        """ Rules for wireframe
        - Each false junction have only one line
        - Each Proper junction have maximum:
            1 Wall line
            2 of any other line
        - Any junction may only have 3 lines
        Trim every subgraph that is biconnected when adding edge between any false junctions?
        """
        for j_idx, node in enumerate(G.nodes()):
            l_idx = np.array([edge2idx[frozenset(e)] for e in G.edges(nbunch=node, data=False)])
            # print(self.lm.get_junction_labels()[j_labels[j_idx]], 'idx', j_idx, 'has', len(l_idx), 'edges')
            if len(l_idx) <= 1:
                continue
            j_label = self.lm.get_junction_labels()[j_labels[j_idx]]
            l_score = score[l_idx]

            if j_label == 'false':
                remove_mask[l_idx] |= l_score < l_score.max()
            #     # print('A remove', l_idx[l_score < l_score.max()])
            elif j_label == 'proper':
                wall_mask = labels[l_idx]==self.lm.get_line_labels().index('wall')
                wall_idx = l_idx[wall_mask]
                if len(wall_idx) > 1:
                    wall_score = l_score[wall_mask]
                    remove_mask[wall_idx] |= wall_score < wall_score.max()
                    # print('B remove', wall_idx[wall_score < wall_score.max()])
                other_idx = l_idx[~wall_mask]
                if len(other_idx) > 2:
                    other_score = l_score[~wall_mask]
                    s_idx = np.argsort(-other_score)
                    remove_idx = l_idx[s_idx[2:]]
                    # print('C remove', remove_idx)
                    remove_mask[remove_idx] = True
            l_idx_mask = ~remove_mask[l_idx]
            l_idx = l_idx[l_idx_mask]
            l_score = l_score[l_idx_mask]
            if len(l_idx) > 3:
                s_idx = np.argsort(-l_score)
                remove_idx = l_idx[s_idx[3:]]
                remove_mask[remove_idx] = True
                # print('D remove', remove_idx)

        if ax:
            from copy import deepcopy
            output_removed = deepcopy(output)
            self._nms_remove(~remove_mask, output_removed)
            self._nms_remove(remove_mask, output)
            self.img_viz._ax_plot_final_lj(ax[0], output, -1, True, show_legend=True, junction_text=True, line_text=True)
            self.img_viz._ax_plot_final_lj(ax[1], output_removed, -1, True, show_legend=True,line_text=True)
        elif np.any(remove_mask):
            self._nms_remove(remove_mask, output)


    def run_line_nms(self, output_list, ax = None):
        if self.line_nms_threshold2 <= 0 and not self.graph_nms:
            return
        if hasattr(output_list, 'keys'):
            output_list = [output_list]

        for output in output_list:
            if self.line_nms_threshold2 > 0:
                self._nms_distance(output)

            if self.graph_nms:# Form the graph to get line indices for each junction.
                self._nms_graph(output, ax=ax)

    def run_plane_nms(self, output_list, ax = None):
        if self.plane_nms_threshold <= 0:
            return
        if hasattr(output_list, 'keys'):
            output_list = [output_list]

        for output in output_list:
            self._nms_distance(output)

    def _plane_nms_iou(self, output):
        juncs_pred = np.array(res['juncs_pred'],dtype=np.float32)
        planes = [np.array(juncs_pred[p]) for p in res['planes_pred']]
        pred_labels = np.array(res['planes_label'])
        scores = np.array(res['planes_label_score'])

    def inference_stats(self, inference_times, epoch):
        median_time = np.median(inference_times)
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        plt.figure()
        plt.boxplot(inference_times, labels = [''], showfliers=False)
        plt.ylabel('Time [seconds]')
        plt.title(f'Median: {median_time:0.2f}, Mean : {mean_time:0.2f}, Std: {std_time:0.3f}')
        fig_path = osp.join(self.plot_dir, 'E{:02}_inference_time_box.pdf'.format(epoch))
        plt.savefig(fig_path)
        plt.close()
        plt.figure()
        plt.hist(inference_times)
        plt.xlabel('Time [seconds]')
        plt.title(f'Median: {median_time:0.2f}, Mean : {mean_time:0.2f}, Std: {std_time:0.3f}')
        fig_path = osp.join(self.plot_dir, 'E{:02}_inference_time_hist.pdf'.format(epoch))
        plt.savefig(fig_path)
        plt.close()


    def eval_sap(self, results, annotations_dict, epoch):
        thresholds = [5,10,15]
        rcs, pcs, sAP = me.evalulate_sap(results, annotations_dict, thresholds, self.lm.get_line_labels())
        self._make_AP_latex_table(sAP, 'sAP', epoch)
        for m_type, thres_dict in sAP.items():
            for t in thres_dict:
                try:
                    fig = self.img_viz.plot_ap(rcs[m_type][t], pcs[m_type][t], sAP[m_type][t], t, AP_string = fr'\mathrm{{sAP}}')
                    fig_path = osp.join(self.plot_dir, 'E{:02}_sAP_{}_{}.pdf'.format(epoch, m_type, t))
                    plt.savefig(fig_path)
                except KeyError:
                    fig = None

                if self.tb_logger:
                    for ap_type, ap_score in sAP[m_type][t].items():
                        self.tb_logger.add_scalar('sAP{} - {} - {}'.format(m_type, t, ap_type), ap_score, global_step = epoch)
                    if fig:
                        self.tb_logger.add_figure('PR lines - {} - {}'.format(m_type,t), fig, global_step = epoch, close=True)

                if self.wandb_run:
                    for ap_type, ap_score in sAP[m_type][t].items():
                        self.wandb_run.log({'sAP{} - {} - {}'.format(m_type, t, ap_type): ap_score,
                                            'epoch': epoch})
                    if fig:
                        fig.canvas.draw()
                        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        self.wandb_run.log({'PR lines - {} - {}'.format(m_type,t): wandb.Image(image_from_plot),
                                            'epoch': epoch})
        return sAP

    def eval_jap(self, results, annotations_dict, epoch):
        thresholds = [0.5,1.0,2.0]
        rcs, pcs, jAP = me.evalulate_jap(results, annotations_dict, thresholds, self.lm.get_junction_labels())
        ap_str = {'valid': r'\mathrm{{j}}_1\mathrm{{AP}}',
                  'label': r'\mathrm{{j}}_3\mathrm{{AP}}',
                  'label_line_valid': r'\mathrm{{j}}_2\mathrm{{AP}}'}
        self._make_AP_latex_table(jAP, 'jAP')
        for m_type, thres_dict in jAP.items():
            dstr = ap_str[m_type]
            for t in thres_dict:
                try:
                    fig = self.img_viz.plot_ap(rcs[m_type][t], pcs[m_type][t], jAP[m_type][t], t, AP_string = dstr)
                    fig_path = osp.join(self.plot_dir, 'E{:02}_jAP_{}_{}.pdf'.format(epoch, m_type, t))
                    plt.savefig(fig_path)
                except KeyError:
                    fig = None

                if self.tb_logger:
                    for ap_type, ap_score in jAP[m_type][t].items():
                        self.tb_logger.add_scalar('jAP{} - {} - {}'.format(m_type, t, ap_type), ap_score, global_step = epoch)
                    if fig:
                        self.tb_logger.add_figure('PR junctions - {} - {}'.format(m_type,t), fig, global_step = epoch, close=True)

                if self.wandb_run:
                    for ap_type, ap_score in jAP[m_type][t].items():
                        self.wandb_run.log({'jAP{} - {} - {}'.format(m_type, t, ap_type): ap_score,
                                            'epoch': epoch})
                    if fig:
                        fig.canvas.draw()
                        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        self.wandb_run.log({'PR junctions - {} - {}'.format(m_type,t): wandb.Image(image_from_plot),
                                            'epoch': epoch})

        return jAP

    def eval_pcap(self, results, annotations_dict, epoch):
        thresholds = [1.0,2.0,100.0]
        rcs, pcs, pcAP = me.evalulate_pcap(results, annotations_dict, thresholds, self.plane_labels)
        # self._make_AP_latex_table(pcAP, 'pcAP')
        for t in thresholds:
            try:
                fig = self.img_viz.plot_ap(rcs[t], pcs[t], pcAP[t], t, AP_string =  r'\mathrm{{pc}}\mathrm{{AP}}')
                fig_path = osp.join(self.plot_dir, 'E{:02}_pcAP_{}.pdf'.format(epoch, t))
                plt.savefig(fig_path)
            except KeyError:
                fig = None

            if self.tb_logger:
                for ap_type, ap_score in pcAP[t].items():
                    self.tb_logger.add_scalar('pcAP{} - {}'.format(t, ap_type), ap_score, global_step = epoch)
                if fig:
                    self.tb_logger.add_figure('PR plane centroids - {}'.format(t), fig, global_step = epoch, close=True)

            if self.wandb_run:
                for ap_type, ap_score in pcAP[t].items():
                    self.wandb_run.log({'pcAP{} - {}'.format(t, ap_type): ap_score,
                                        'epoch': epoch})
                if fig:
                    fig.canvas.draw()
                    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.wandb_run.log({'PR plane centroids - {}'.format(t): wandb.Image(image_from_plot),
                                        'epoch': epoch})

        return pcAP

    def eval_pap(self, results, annotations_dict, epoch):
        thresholds = np.linspace(0.5,1.0,11) #Instead of arange(0.5,1.0,0.05)
        rcs, pcs, pAP = me.evalulate_pap(results, annotations_dict, thresholds, self.plane_labels)

        plot_thres = [0.5, 0.9, 'mean']
        pAP_table = {t:pAP[t] for t in plot_thres}
        self._make_AP_latex_table({'pAP':pAP_table}, 'pAP',epoch)
        for t in plot_thres:
            try:
                fig = self.img_viz.plot_ap(rcs[t], pcs[t], pAP[t], t, AP_string = r'\mathrm{{pAP}}')
                fig_path = osp.join(self.plot_dir, 'E{:02}_pAP_{}.pdf'.format(epoch, t))
                plt.savefig(fig_path)
            except KeyError:
                fig = None

            if self.tb_logger:
                for ap_type, ap_score in pAP[t].items():
                    self.tb_logger.add_scalar('pAP{} - {}'.format(t, ap_type), ap_score, global_step = epoch)
                if fig:
                    self.tb_logger.add_figure('PR planes - {}'.format(t), fig, global_step = epoch, close=True)

            if self.wandb_run:
                for ap_type, ap_score in pAP[t].items():
                    self.wandb_run.log({'pAP{} - {}'.format(t, ap_type): ap_score,
                                        'epoch': epoch})
                if fig:
                    fig.canvas.draw()
                    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.wandb_run.log({'PR planes - {}'.format(t): wandb.Image(image_from_plot),
                                        'epoch': epoch})

        return pAP

    def eval_lsun_kp(self, results, annotations_dict, epoch):
        thresholds, kp_error = me.evalulate_lsun_kp(results, annotations_dict)
        fig = self.img_viz.plot_lsun_kp(thresholds,  kp_error)
        fig_path = osp.join(self.plot_dir, 'E{:02}_LSUN_KP.png'.format(epoch))
        plt.savefig(fig_path)

        t_idx = np.argmin(np.abs(thresholds - 0.5))
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('LSUN_KP_{}'.format(thresholds[t_idx]), kp_error[t_idx], global_step = epoch)
            self.tb_logger.add_figure('LSUN KP', fig, global_step = epoch, close=True)

        if self.wandb_run:
            self.wandb_run.log({'LSUN_KP_{}'.format(thresholds[t_idx]): kp_error[t_idx],
                                    'epoch': epoch})
            if fig:
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                self.wandb_run.log({'LSUN KP': wandb.Image(image_from_plot),
                                    'epoch': epoch})

    def eval_room_layout(self, results, annotations_dict, epoch):
        thresholds = [0.1,0.5]
        room_result = me.evalulate_room_layout(results, annotations_dict, thresholds, self.plane_labels)
        # room_result_rev = {}
        # for t, metric_dict in room_result.items():
        #     for m, val in metric_dict.items():
        #         if m in room_result_rev:
        #             room_result_rev[m][t] = val
        #         else:
        #             room_result_rev[m] = {t:val}
        room_result_table = {t:{} for t in room_result.keys()}
        for t, metric_dict in room_result.items():
            for metric, value in metric_dict.items():
                if metric.startswith('PE') or  metric.startswith('iou'):
                    room_result_table[t][metric] = value
        self._make_AP_latex_table({'room':room_result_table}, 'room',epoch)

        if self.wandb_run:
            for t, t_res in room_result.items():
                for metric, value in t_res.items():
                    try:
                        self.wandb_run.log({f'room_{metric}_t{t}': float(value),
                                            'epoch': epoch})
                    except TypeError:
                        pass
                fig = self.img_viz.plot_plane_parameter_residuals(t_res['parameter_hist'], t_res['parameter_bin_edges'])
                self._log_fig(fig, f'Plane parameter residuals T:{t}', epoch)


    def test_all(self, model_graph = False, subsample = 1):
        for epoch, model in self._get_all_models(subsample=subsample):
            self.test_model(epoch = epoch, model = model, model_graph = model_graph)


    def test_filenames(self, filenames, model=None):
        filenames = set(filenames)
        if multi_gpu.num_gpus() > 1:
            raise NotImplementedError('test_filenames not adapted for MultiGPU')
        if model is None:
            model = self._load_model_from_cfg()
        model.eval()

        for name, dataset in self.datasets:

            self.logger.info('Finding filenames in {} dataset'.format(name))

            for i, (images, annotations) in enumerate(tqdm(dataset)):
                if len(filenames) == 0:
                    break

                ann = annotations if hasattr(annotations,'keys') else annotations[0]
                im_name = ann['filename']
                if not im_name in filenames:
                    continue
                else:
                    filenames.remove(im_name)

                with torch.no_grad():
                    output, extra_info = model(to_device(images, self.device), to_device(annotations, self.device))
                    output = to_device(output,'cpu')

                    im = np.array(dataset.dataset.image(i))
                    self.run_line_nms(output) #TMP
                    self._tb_plots(0, im, im_name, output, None)
                    # fig = self.img_viz.plot_pred(im, im_name, output, None,
                    #                              score_threshold = self.score_threshold,
                    #                              ignore_invalid_junc = True,
                    #                              show_legend=self.show_legend)
                    # fname = '{}_pred.pdf'.format(osp.splitext(im_name)[0])
                    # plt.savefig(osp.join(self.plot_dir, fname))

    def test_model_on_folder(self, folder_path, model = None):
        if multi_gpu.num_gpus() > 1:
            raise NotImplementedError('test_model_on_folder not adapted for MultiGPU')
        if model is None:
            model = self._load_model_from_cfg()
        model.eval()
        img_ext = set(['.png', '.jpg', '.jpeg'])

        img_paths = []
        for fn in os.listdir(folder_path):
            if osp.splitext(fn)[1].lower() in img_ext:
                img_paths.append(osp.join(folder_path, fn))

        self.logger.info('Testing on folder {}'.format(folder_path))
        for i, img_path in enumerate(tqdm(img_paths)):
            image_int = io.imread(img_path)
            image_tensor = self.img_transform(image_int.astype(float)[:,:,:3])
            filename = osp.basename(img_path)
            ann = {
                'height': image_int.shape[0],
                'width': image_int.shape[1],
                'filename': filename
            }

            with torch.no_grad():
                output, extra_info = model(image_tensor[None].to(self.device), [ann])
                output = to_device(output,'cpu')
                self.run_line_nms(output)

            self._tb_plots(0, image_int, filename, output, None)

    def test_model(self, model = None, epoch = 0, model_graph = False):
        if model is None:
            model = self._load_model_from_cfg()
        model.eval()
        plotted_scenes = set()

        inference_times = []
        ctimer = CudaTimer()

        if model_graph:
            self._tb_graph(model)

        for name, dataset in self.datasets:
            results = []
            annotations_dict = {}
            self.logger.info('Testing on {} dataset'.format(name))

            # Synchronize start for each dataset so that queue does not fill up.
            multi_gpu.barrier()
            tqdm_dataset = tqdm(dataset) if multi_gpu.is_main_proc() else dataset
            for i, (images, annotations) in enumerate(tqdm_dataset):

                ann = annotations if hasattr(annotations,'keys') else annotations[0]
                try:
                    scene_id = int(ann['filename'][1:6])
                except ValueError:
                    scene_id = len(plotted_scenes)
                annotations_dict[ann['filename']] = ann
                with torch.no_grad():
                    ctimer.start_timer()
                    output, extra_info = model(to_device(images, self.device), to_device(annotations, self.device))
                    inference_times.append(ctimer.end_timer())
                    output = to_device(output,'cpu')

                t = time.time()
                if multi_gpu.is_main_proc():
                    results.append(self._to_list(output))
                    self._get_all_queued(results)
                else:
                    self._queue(self._to_list(output))

                if multi_gpu.is_main_proc() and len(plotted_scenes) < self.nbr_plots and scene_id not in plotted_scenes:
                    im = dataset.dataset.image(filename=ann['filename'])
                    extra_info = to_device(extra_info,'cpu')
                    output = to_device(output,'cpu')
                    #------------DBG
                    # fig1, ax1 = self.img_viz.no_border_imshow(np.array(im))
                    # fig2, ax2 = self.img_viz.no_border_imshow(np.array(im))
                    # self.run_line_nms(output, ax = [ax1, ax2]) #TMP
                    # title1 = 'Filtered_{}'.format(ann['filename'])
                    # title2 = 'Removed_{}'.format(ann['filename'])
                    # ax1.set_title(title1)
                    # ax2.set_title(title2)
                    # fig1.savefig(osp.join('/host_home/debug/SRW_NMS',title1))
                    # fig2.savefig(osp.join('/host_home/debug/SRW_NMS',title2))
                    #------------DBG END
                    self.run_line_nms(output) #TMP
                    self._tb_plots(epoch, im, ann['filename'], output, extra_info, idx=len(plotted_scenes))
                    plotted_scenes.add(scene_id)


            if multi_gpu.is_main_proc():
                self._wait_for_queue_finish(results)
            else:
                self._queue_mark_dataset_finished()
            multi_gpu.barrier()

            # Only main process calculates score and save accumulated result
            if not multi_gpu.is_main_proc():
                continue

            outpath_dataset = osp.join(self.output_dir,'{}_epoch{:05d}.json'.format(name, epoch))
            self.logger.info('Writing the results of the {} dataset into {}'.format(name,
                        outpath_dataset))
            with open(outpath_dataset,'w') as _out:
                json.dump(results,_out)

            # Get all annotations here so that the main process has all annotations
            annotations_dict = {ann['filename']: ann for ann in to_device(dataset.dataset.annotations,'cpu')}


            if self.calculate_sap:
                self.run_line_nms(results)
                self.eval_sap(results, annotations_dict, epoch)
                self.eval_jap(results, annotations_dict, epoch)
                self.eval_lsun_kp(results, annotations_dict, epoch)
            self.eval_pap(results, annotations_dict, epoch)
            if 'plane_parameters' in results[0]:
                self.eval_room_layout(results, annotations_dict, epoch)
            self.eval_pcap(results, annotations_dict, epoch)
            self.inference_stats(inference_times, epoch)



    def _to_list(self, output):
        new_output = {}
        for k in output.keys():
            if isinstance(output[k], torch.Tensor):
                new_output[k] = output[k].tolist()
            elif isinstance(output[k], list) and len(output[k]) > 0 and isinstance(output[k][0], torch.Tensor):
                new_output[k] = [v.tolist() for v in output[k]]
            else:
                new_output[k] = output[k]
        return new_output

    def _queue_mark_dataset_finished(self):
        self.result_queue.put(POISON_PILL)

    def _queue(self, output):
        self.result_queue.put(output)

    # TODO: Timeout?
    def _wait_for_queue_finish(self, all_result):
        if not multi_gpu.is_multiprocessing():
            return

        while self.nbr_poision_pills + 1 < multi_gpu.num_gpus():
            self._get_all_queued(all_result)
            time.sleep(0.1)

        # Reset for next dataset
        self.nbr_poision_pills = 0

    def _get_all_queued(self, all_result):
        if not multi_gpu.is_multiprocessing():
            return

        while True:
            try:
                r = self.result_queue.get_nowait()
            except queue.Empty:
                return

            if r == POISON_PILL:
                self.nbr_poision_pills += 1
            else:
                all_result.append(r)

def test(rank, cfg, args, output_dir, result_queue = None):

    multi_gpu.setup(rank,cfg.SOLVER.NUM_GPUS)
    logger = multi_gpu.setup_logger(output_dir, datetime.now().isoformat(timespec='seconds'))

    tester = ModelTester(cfg,
                         output_dir = output_dir,
                         validation = args.val,
                         nbr_plots = args.nbr_plots,
                         write_tb =not args.skip_tb and not args.img_folder,
                         write_latex = args.write_latex,
                         show_legend = not args.no_legend,
                         disable_wandb = args.disable_wandb,
                         resume_wandb = args.resume_wandb,
                         result_queue = result_queue)
    if args.img_folder:
        tester.test_model_on_folder(args.img_folder)
    elif args.filenames:
        with open(args.filenames, 'r') as f:
            filenames = set(json.load(f))
        tester.test_filenames(filenames)
    elif args.all:
        tester.test_all(model_graph=args.graph, subsample = args.subsample)
    else:
        tester.test_model(model_graph=args.graph)

    multi_gpu.barrier()
    multi_gpu.cleanup()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HAWP Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True,
                        )

    parser.add_argument("--nbr-plots",
                        default=5,
                        type=int,
                        help='Number of plots to show')

    parser.add_argument("--img-folder",
                        default=None,
                        type=str,
                        help='Image folder to run model on. Overrides dataset choice in config.')

    parser.add_argument("--filenames",
                        default=None,
                        type=str,
                        help='JSON file with filenames from the dataset to run test on, plots all predictions.')

    parser.add_argument("--all",
                        default=False,
                        action='store_true',
                        help='Test all model checkpoints')

    parser.add_argument("--subsample",
                        default=1,
                        type=int,
                        help='Subsample checkpoints if testing all')

    parser.add_argument("--val",
                        default=False,
                        action='store_true',
                        help='Use validation data')

    parser.add_argument("--graph",
                        default=False,
                        action='store_true')
    parser.add_argument("--disable-wandb",
                        action='store_true')
    parser.add_argument("--resume-wandb",
                        action='store_true')
    parser.add_argument('-t', '--skip-tb', action='store_true', help='Skip logging to tensorboard')
    parser.add_argument('-l', '--write-latex', action='store_true', help='Make LaTeX tables')
    parser.add_argument('--no-legend', action='store_true', help='Skip legend ')
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    args = parser.parse_args()

    from parsing.config import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.val:
        output_dir = osp.join(cfg.OUTPUT_DIR, 'val')
    else:
        output_dir = osp.join(cfg.OUTPUT_DIR, 'test')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger('srw-test', output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # Will move datasets if DATASET.TMP_DIR is specified in the config
    move_datasets(cfg, logger)
    multi_gpu.setup_context()
    result_queue = ModelTester.setup_queue()
    test_args = (cfg, args, output_dir, result_queue)
    if cfg.SOLVER.NUM_GPUS > 1:

        mp.spawn(
        test,
        args=test_args,
        nprocs=cfg.SOLVER.NUM_GPUS
        )
    else:
        test(0, *test_args)
