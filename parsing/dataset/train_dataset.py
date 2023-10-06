import torch
from torch.utils.data import Dataset

import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy
import os
class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform = None, vflip = False, hflip = False):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)

        # Make sure there is at least one negative edge for each image
        self.annotations = [a for a in self.annotations if a['edges_negative']]
        self.transform = transform
        self.vflip = vflip
        self.hflip = hflip

        ann_folder = osp.dirname(ann_file)
        self.dbg_dir_prior = osp.join(ann_folder, 'train_dbg', 'prior_transform')
        self.dbg_dir_post = osp.join(ann_folder, 'train_dbg', 'post_transform')
        os.makedirs(self.dbg_dir_prior, exist_ok=True)
        os.makedirs(self.dbg_dir_post, exist_ok=True)


    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        for key,_type in (['junctions',np.float32],
                          ['junctions_semantic',np.int32],
                          ['edges_positive',np.int32],
                          ['edges_negative',np.int32],
                          ['edges_semantic',np.int32],
                          ['camera_pose',np.float32]):

            ann[key] = np.array(ann[key],dtype=_type)

        for plane_type in ['planes', 'planes_negative']:
            for plane in ann[plane_type]:
                for key,_type in (['junction_idx',np.int32],
                                  ['edge_idx',np.int32],
                                  ['semantic',np.int32],
                                  ['centroid',np.float32]):
                    plane[key] = np.array(plane[key],dtype=_type)

        for plane in ann['planes']:
            plane['parameters'] = np.array(plane['parameters'],dtype=np.float32)

        if not 'junctions_semantic' in ann:
            ann['junc_occluded'] = np.array(ann['junc_occluded'],dtype=np.bool)

        assert np.all(ann['edges_semantic'] > 0) #Assumes that the dataset also has an invalid class as 0

        width = ann['width']
        height = ann['height']
        #Randomize flip
        if self.hflip and random.getrandbits(1):
            image = image[:,::-1,:]
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
            for plane_type in ['planes', 'planes_negative']:
                for plane in ann[plane_type]:
                    plane['centroid'][0] = width-plane['centroid'][0]
        if self.vflip and random.getrandbits(1):
            image = image[::-1,:,:]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
            for plane_type in ['planes', 'planes_negative']:
                for plane in ann[plane_type]:
                    plane['centroid'][1] = height-plane['centroid'][1]


        if self.transform is not None:
            image, ann = self.transform(image,ann)

        return image, ann

    def __len__(self):
        return len(self.annotations)

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
