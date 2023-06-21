import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import os.path as osp
import numpy as np
import torch
import seaborn as sns

def set_fonts():
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

class ImagePlotter:
    def __init__(self, line_classes, junction_classes, plane_classes = []):
        self.JUNCTION_CLASSES = junction_classes
        self.JUNCTION_COLORS = sns.color_palette('bright',len(self.JUNCTION_CLASSES))
        self.JUNCTION_MARKERS = ['x', 'o', 'd']
        self.JUNCTION_SIZE = 10
        self.LINE_WIDTH = 5
        self.JUNCTION_LEGEND = [Line2D([0],[0], label=l, color=c, marker='o') for (l,c) in zip(self.JUNCTION_CLASSES, self.JUNCTION_COLORS)]
        self.SINGLE_LINE_CLASSES = line_classes
        self.LINE_COLORS = sns.color_palette('bright',len(self.SINGLE_LINE_CLASSES))
        self.LINE_LEGEND = [Line2D([0],[0], label=l, color=c) for (l,c) in zip(self.SINGLE_LINE_CLASSES, self.LINE_COLORS)]
        self.PLANE_CLASSES = plane_classes
        if plane_classes:
            self.PLANE_COLORS = sns.color_palette('bright',len(self.PLANE_CLASSES))
            self.PLANE_ALPHA = 0.2
            self.HATCH_PLANE_ALPHA = 0.8
            self.PLANE_LINESTYLE = None
            self.PLANE_LINEWIDTH = 4
            self.PLANE_HATCHES = ["/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
            self.PLANE_HATCH_COUNTER = 0
        set_fonts()

    def _add_polygon_patch(self, ax, poly, poly_color, skip_hatch = False):
        if not skip_hatch:
            hatch = self.PLANE_HATCHES[self.PLANE_HATCH_COUNTER]
            self.PLANE_HATCH_COUNTER += 1
            self.PLANE_HATCH_COUNTER = self.PLANE_HATCH_COUNTER % len(self.PLANE_HATCHES)
        else:
            hatch = None
        ax.add_patch(Polygon(poly,
                             facecolor = poly_color,
                             edgecolor = poly_color,
                             alpha=self.PLANE_ALPHA if skip_hatch else self.HATCH_PLANE_ALPHA,
                             linewidth = self.PLANE_LINEWIDTH,
                             linestyle = self.PLANE_LINESTYLE,
                             fill=skip_hatch,
                             hatch = hatch,
                             closed=True
                             ))


    def _plot_plane_legend(self, ax=None, plot_legend = True):
        handles = [Line2D([0],[0], color=self.PLANE_COLORS[i], markersize=10, linestyle='none', marker='s', alpha=self.PLANE_ALPHA) for i in range(len(self.PLANE_CLASSES))]
        labels = list(self.PLANE_CLASSES)
        if plot_legend:
            ax = plt.gca() if not ax else ax
            ax.legend(handles, labels, shadow=True, fontsize = 'large')
        return handles, labels

    def _plot_legend(self, edges_semantic, ax=None):
        slabels = np.unique(edges_semantic)
        plane_handles, plane_labels = self._plot_plane_legend(plot_legend = False)
        handles = [
            Line2D([0],[0], color=self.LINE_COLORS[i], linewidth=self.LINE_WIDTH) for i in slabels
            ] + [
            Line2D([0],[0], color=self.JUNCTION_COLORS[i], markersize=self.JUNCTION_SIZE, linestyle='none', marker=self.JUNCTION_MARKERS[i]) for i in range(1,len(self.JUNCTION_CLASSES))
            ] + plane_handles
        labels = [self.SINGLE_LINE_CLASSES[i] for i in slabels] + list(self.JUNCTION_CLASSES)[1:] + plane_labels
        if not ax:
            ax = plt.gca()
        ax.legend(handles, labels, shadow=True, fontsize = 'large')

    def plot_gt_image(self, img, ann, out_folder, desc = None, use_negative_edges = False, use_negative_planes = False,edges_text = None, show_legend=True, ext='.png', skip_hatch = False):

        if isinstance(img, torch.Tensor):
            img = np.rollaxis(img.numpy(), 0, 3)
        else:
            img = img.astype(int)

        if isinstance(ann['junctions'], (list, tuple)):
            junctions = np.array(ann['junctions'])
        else:
            junctions = ann['junctions']


        if use_negative_edges:
            edges = ann['edges_negative']
            edges_semantic = [0]*len(edges)
        else:
            edges = ann['edges_positive']
            edges_semantic = ann['edges_semantic']

        if edges_text is None:
            edges_text = [None]*len(edges)

        dpi = 100
        fig, ax = self.no_border_imshow(img,dpi=dpi)
        for edge, label, text in zip(edges, edges_semantic, edges_text):
            ax.plot(junctions[edge,0], junctions[edge,1], color=self.LINE_COLORS[label], linewidth=self.LINE_WIDTH)
            if text:
                ax.text(*junctions[edge].mean(axis=0), text, color=self.JUNCTION_COLORS[0])
        if show_legend:
            self._plot_legend(edges_semantic, ax=ax)
        if 'junctions_semantic' in ann:
            for junc, jsem in zip(junctions, ann['junctions_semantic']):
                ax.plot(*junc, marker=self.JUNCTION_MARKERS[jsem], color=self.JUNCTION_COLORS[jsem], markersize=self.JUNCTION_SIZE)
        else:
            for junc, occ in zip(junctions, ann['junc_occluded']):
                ax.plot(*junc, marker=self.JUNCTION_MARKERS[2-occ], color=self.JUNCTION_COLORS[2-occ],markersize=self.JUNCTION_SIZE)

        if self.PLANE_CLASSES and 'planes' in ann:
            plane_list = ann['planes_negative'] if use_negative_planes else ann['planes']
            neg_colors = sns.color_palette("husl", len(ann['planes_negative']))
            for idx, plane in enumerate(plane_list):
                poly = junctions[np.array(plane['junction_idx'],dtype=int)]
                if use_negative_planes:
                    poly_color = neg_colors[idx]
                else:
                    poly_color = self.PLANE_COLORS[plane['semantic']] if plane['junction_idx'][0] == plane['junction_idx'][-1] else 'k'
                self._add_polygon_patch(ax, poly, poly_color, skip_hatch = skip_hatch)
                # ax.plot(*plane['centroid'], marker='s', color=self.PLANE_COLORS[plane['semantic']], markersize=self.JUNCTION_SIZE)

        # print('plane_centroids' in ann)
        # if self.PLANE_CLASSES and 'plane_centroids' in ann:
        #     print('PLotting', len(ann['plane_centroids']))
        #     for pc, plane in zip(ann['plane_centroids'], ann['planes']):


        fname = osp.splitext(ann['filename'])[0]
        new_fname = '{}_{}{}'.format(fname, desc, ext)
        plt.savefig(osp.join(out_folder, new_fname))
        plt.close()

    def _ax_plot_planes(self, ax, model_output, score_threshold, skip_hatch = False):
        planes = [p.numpy() for p in model_output['planes_pred']]
        planes_labels = model_output['planes_label']
        planes_scores = model_output['planes_label_score']
        p_mask = planes_scores > score_threshold
        junctions = model_output['juncs_pred'].numpy()

        unique_labels = np.unique(planes_labels)
        for l in unique_labels:
            if l == 0:
                continue

            tmp_idx = np.flatnonzero(p_mask & (planes_labels==l))
            c = self.PLANE_COLORS[l]
            for p_idx in tmp_idx:
                poly = junctions[np.array(planes[p_idx],dtype=np.int)]
                self._add_polygon_patch(ax, poly, c,skip_hatch=skip_hatch)

    def _ax_plot_planes_centroids(self, ax, model_output, score_threshold):
        pcentroid = model_output['pcentroid_pred'].numpy()

        pcentroid_labels = model_output['pcentroid_label']
        pcentroid_scores = model_output['pcentroid_label_score']
        p_mask = pcentroid_scores > score_threshold

        unique_labels = np.unique(pcentroid_labels)
        for l in unique_labels:
            if l == 0:
                continue

            tmp_mask = p_mask & (pcentroid_labels==l)
            ax.plot(*pcentroid[tmp_mask].T, marker='s',linestyle='none',
                    color = self.PLANE_COLORS[l], markersize=self.JUNCTION_SIZE)

    def _ax_plot_final_lj(self, ax, model_output, score_threshold, ignore_invalid_junc, show_legend = True, junction_text = False, line_text = False):
        lines = model_output['lines_pred'].numpy()
        lines_scores = model_output['lines_label_score'].numpy()
        lines_label = model_output['lines_label'].numpy()
        l2j_idx = model_output['edges_pred'].numpy()
        l_mask = lines_scores > score_threshold
        junctions = model_output['juncs_pred'].numpy()
        if ignore_invalid_junc and 'juncs_score' in model_output:
            junctions_all_scores = model_output['juncs_score'].numpy()
            junctions_label = model_output['juncs_label'].numpy()
            invalid_mask = junctions_label == 0
            junctions_label[invalid_mask] = np.argmax(junctions_all_scores[invalid_mask,1:], axis=1) + 1
        else:
            junctions_label = model_output['juncs_label'].numpy()

        # ax.plot(lines[l_mask,::2],lines[l_mask,1::2], 'r-')
        j_mask = np.zeros(junctions_label.shape, dtype=bool)
        unique_labels = np.unique(lines_label)
        for l in unique_labels:
            if l == 0:
                continue

            tmp_mask  = l_mask & (lines_label==l)
            ax.plot([lines[tmp_mask,0], lines[tmp_mask,2]],
                    [lines[tmp_mask,1], lines[tmp_mask,3]],
                    color = self.LINE_COLORS[l],
                    linewidth = self.LINE_WIDTH)
            jidx = np.unique(l2j_idx[tmp_mask])
            j_mask[jidx] = True
            if line_text:
                for lidx, lpos in zip(np.flatnonzero(tmp_mask), lines[tmp_mask]):
                    ax.text(*(lpos[:2]+lpos[2:])/2, f'E{lidx}', color='white')
        for l in range(len(self.JUNCTION_CLASSES)):
            tmp_mask  = j_mask & (junctions_label==l)
            ax.plot(*junctions[tmp_mask].T, marker=self.JUNCTION_MARKERS[l],linestyle='none',
                    color = self.JUNCTION_COLORS[l], markersize=self.JUNCTION_SIZE)
        if junction_text:
            for jidx, jpos in enumerate(junctions):
                ax.text(*jpos, f'J{jidx}', color='white')
        if show_legend:
            self._plot_legend(unique_labels, ax = ax)

    def plot_final_pred(self, img, img_name, model_output, score_threshold = 0.9, ignore_invalid_junc = False, show_legend = True, skip_lines = False, skip_planes=False, skip_hatch = False):
        fig, ax = self.no_border_imshow(img)
        if len(model_output['lines_pred']) > 0 and not skip_lines:
            self._ax_plot_final_lj(ax, model_output, score_threshold, ignore_invalid_junc, show_legend=show_legend)
        if len(model_output['planes_pred']) > 0 and not skip_planes:
            self._ax_plot_planes(ax, model_output, score_threshold, skip_hatch = skip_hatch)
        # if len(model_output.get('pcentroid_pred',[])) > 0 and not skip_planes:
        #     self._ax_plot_planes_centroids(ax, model_output, score_threshold)
        plt.title('{}, T={}'.format(img_name, score_threshold))
        return fig

    def plot_pred_top_planes(self, img, img_name, model_output, show_legend = True):
        # TODO: Make input parameter
        nbr_top = 11
        fig, axes = plt.subplots(3,4,sharex = True, sharey = True, dpi = 100, figsize=(16,9))


        if len(model_output['planes_pred']) == 0:
            return fig
        planes_labels = model_output['planes_label']
        #Remove background planes
        keep_idx = np.flatnonzero(planes_labels > 0)
        planes = [model_output['planes_pred'][p_idx].numpy() for p_idx in keep_idx]
        planes_scores = model_output['planes_label_score'][keep_idx]
        planes_labels = planes_labels[keep_idx]
        _, top_p_idx = planes_scores.topk(min(nbr_top, len(planes_scores)))
        junctions = model_output['juncs_pred'].numpy()

        for ax_idx, p_idx in enumerate(top_p_idx):
            ax = axes.flat[ax_idx]
            ax.imshow(img)
            ax.axis('off')

            c = self.PLANE_COLORS[planes_labels[p_idx]]
            poly = junctions[np.array(planes[p_idx],dtype=np.int)]
            self._add_polygon_patch(ax, poly, c, skip_hatch = True)
            ax.set_title(f'Score: {planes_scores[p_idx]:0.2f}')


        axes.flat[nbr_top].axis('off')
        self._plot_plane_legend(ax=axes.flat[nbr_top])
        plt.subplots_adjust(wspace=0.01, hspace=0, left=0.01, right=0.99, top=0.95, bottom=0.01)

        return fig


    def plot_pred(self, img, img_name, model_output, model_extra_info = None, score_threshold = 0.9, ignore_invalid_junc = False, show_legend = True):
        if (not model_extra_info) or ('junc_prior_ver' not in model_extra_info):
            return self.plot_final_pred(img, img_name, model_output, score_threshold, ignore_invalid_junc,show_legend=show_legend)

        try:
            a_ratio = img.shape[1]/img.shape[0]
        except AttributeError:
            a_ratio = img.width/img.height
        h_inch = 5 #per image
        size = (3*h_inch*a_ratio, h_inch)

        fig, axes = plt.subplots(1,4,sharex = True, sharey = True, figsize = size, dpi = 100)
        for ax in axes.flat:
            ax.imshow(img)
            ax.axis('off')
        if len(model_output['lines_pred']) > 0:
            self._ax_plot_final_lj(axes[0], model_output, score_threshold, ignore_invalid_junc,show_legend=show_legend)
        if len(model_output['planes_pred']) > 0:
            self._ax_plot_planes(axes[0], model_output, score_threshold)
        if len(model_output.get('pcentroid_pred',[])) > 0:
            self._ax_plot_planes_centroids(axes[0], model_output, score_threshold)
        axes[0].set_title('{}, T={}'.format(img_name, score_threshold))



        if model_extra_info['lines_prior_scoring'] is not None:
            lines = model_extra_info['lines_prior_scoring'].numpy()
            axes[1].plot([lines[:,0], lines[:,2]],
                        [lines[:,1], lines[:,3]], 'r-')
            N_lines = lines.shape[0]
        else:
            N_lines = 0

        junctions = model_extra_info['junc_prior_ver'].numpy()
        axes[1].plot(*junctions.T,'b.')
        axes[1].set_title('Line Prior Scoring [{}]'.format(N_lines))

        axes[2].plot(*junctions.T,'b.')
        axes[2].set_title('Junction Prop.')

        lines = model_extra_info['lines_prior_ver'].numpy()
        axes[3].plot([lines[::100,0], lines[::100,2]],
                [lines[::100,1], lines[::100,3]], 'r-')
        axes[3].set_title('Line Prop.')
        plt.subplots_adjust(wspace=0.01, hspace=0, left=0.01, right=0.99, top=0.95, bottom=0.01)

        return fig



    def no_border_imshow(self, img, dpi=100, autoscale = False, ax= None):
        if isinstance(img, torch.Tensor):
            img = np.rollaxis(img.to('cpu').numpy(), 0, 3)

        try:
            img_shape = img.shape[:2]
        except AttributeError:
            img_shape = [img.height, img.width]

        if ax is None:
            size = [d/dpi for d in img_shape]
            fig = plt.figure(frameon=False)
            fig.set_size_inches(*size[::-1])
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            fig.add_axes(ax)
        else:
            fig = None
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.imshow(img)
        ax.set_autoscale_on(autoscale)


        return fig, ax


    def plot_plane_parameter_residuals(self, param_hist, param_bins):
        fig, all_ax = plt.subplots(2,2)
        desc = ['nx', 'ny', 'nz', 'd']
        for ax, counts, edges, titlestr in zip(all_ax.flat, param_hist, param_bins, desc):
            ax.bar(edges[:-1], counts, width=np.diff(edges), edgecolor="black", align="edge")
            ax.set_xlabel('Plane parameter residuals')
            ax.set_ylabel('Count')
            ax.set_title(titlestr)
        plt.tight_layout()
        return fig


    def plot_ap(self, rcs, pcs, AP, threshold, desc = '', AP_string = '\mathrm{{AP}}'):
        sns.set_style('whitegrid')
        if 'jAP' in AP_string:
            colors = {l:c for l,c in zip(self.JUNCTION_CLASSES, self.JUNCTION_COLORS)}
            colors['all'] = self.JUNCTION_COLORS[0]
        elif 'sAP' in AP_string:
            colors = {l:c for l,c in zip(self.SINGLE_LINE_CLASSES, self.LINE_COLORS)}
            colors['all'] = self.LINE_COLORS[0]
        else:
            colors = None
        line_style = ['solid', 'dotted', 'dashdot', 'dashed']

        fig = plt.figure()
        mAP = AP.get('mean', AP['all'])
        desc = f'{desc} - ' if desc else ''
        AP_string = fr'{desc}m${AP_string}^{{{threshold}}}$ = {mAP:.1f}'
        f_scores = np.linspace(0.2,0.9,num=8)
        color = sns.color_palette('muted',10)[-1]
        for f_score in f_scores:
            x = np.linspace(0.01,1)
            y = f_score*x/(2*x-f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color=color, alpha=0.3)
            plt.annotate("F={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4,fontsize=9)

        plt.rc('legend',fontsize=10)
        plt.grid(True)
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.gca().set_aspect('equal')
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        for ki, k in enumerate(rcs):
            lname = k if hasattr(k, 'strip') else '-'.join(k)
            color = colors[lname] if colors else None
            plt.plot(rcs[k],pcs[k],label=f'{lname}, AP:{AP[k]:.1f}', linestyle=line_style[ki%4], color=color)
        plt.title(AP_string)
        plt.legend()
        plt.tight_layout()
        return fig

    def plot_lsun_kp(self, thresholds, kp_error):
        sns.set_style('whitegrid')
        fig = plt.figure()
        plt.grid(True)
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.xticks(np.arange(0, 1.0, step=0.1))
        plt.yticks(np.arange(0, 1.0, step=0.1))
        plt.xlabel("Threshold")
        plt.ylabel("LSUN KP error")
        plt.plot(thresholds, kp_error, '.-')
        plt.title('LSUN KP error')
        return fig
