import os
import os.path as osp
import cv2
import matplotlib
# matplotlib.use('Cairo')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numpy as np
import argparse
import yaml
import multiprocessing as mp
import time
from glob import glob

PLANE_LABELS= ['invalid', 'wall','floor','ceiling']


class AnnStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_nbr_junctions = []
        self.all_nbr_planes = {l:[] for l in PLANE_LABELS}
        self.plane_parameters = []

    def add(self, ann):
        for a in ann:
            self.all_nbr_junctions.append(len(a['junctions']))
            p_labels = np.array([p['semantic'] for p in a['planes']])
            for i in range(1,len(PLANE_LABELS)):
                l_name = PLANE_LABELS[i]
                self.all_nbr_planes[l_name].append(np.count_nonzero(i==p_labels))
            for p in a['planes']:
                self.plane_parameters.append(p['parameters'])



    def add_ann_from_path(self, ann_path):
        with open(ann_path) as f:
            ann = yaml.safe_load(f)
        self.add(ann)

    def add_files(self, ann_paths, nbr_workers = 1):
        for a_path in tqdm(ann_paths):
            self.add_ann_from_path(a_path)

    # def _compute_stats(self, ann):


    def write_yaml(self, yaml_path, plot_path):
        plt.figure()
        data = [self.all_nbr_junctions] + [self.all_nbr_planes[l] for l in PLANE_LABELS[1:]]
        data_labels = ['Nbr Junctions'] + PLANE_LABELS[1:]
        for i, (d, dl) in enumerate(zip(data,data_labels)):
            plt.subplot(2,2,i+1)
            plt.hist(d)
            plt.title(dl)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        yaml_dict = {
            'JUNCTION_MEAN': float(np.mean(self.all_nbr_junctions)),
            'JUNCTION_MEDIAN': float(np.median(self.all_nbr_junctions)),
        }
        for l in PLANE_LABELS[1:]:
            yaml_dict.update({
                f'PLANE_{l.upper()}_MEAN': float(np.mean(self.all_nbr_planes[l])),
                f'PLANE_{l.upper()}_MEDIAN': float(np.median(self.all_nbr_planes[l]))
            })

        plane_parameters = np.array(self.plane_parameters)
        plane_mean = plane_parameters.mean(axis=0)
        plane_std = plane_parameters.std(axis=0)
        yaml_dict['PLANE_PARAMETER_MEAN'] = plane_mean.tolist()
        yaml_dict['PLANE_PARAMETER_STD'] = plane_std.tolist()

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_dict, f)
        return yaml_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate stats for annotations')
    parser.add_argument('ann_files', type=str, nargs='+', help='Annotation files')
    parser.add_argument('--out', type=str, default = 'stats.yaml', help='Path to output yaml')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')

    args = parser.parse_args()

    t = time.time()
    stats = AnnStats()
    stats.add_files(args.ann_files, nbr_workers = args.nbr_workers)
    result = stats.write_yaml(args.out, args.out + '.svg')
    print('Took {}s'.format(time.time()-t))
    print(result)
