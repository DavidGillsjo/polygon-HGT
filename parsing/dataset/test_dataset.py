import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import copy
from PIL import Image
from skimage import io
import os
import os.path as osp
import numpy as np
class TestDatasetWithAnnotations(Dataset):
    '''
    Format of the annotation file
    annotations[i] has the following dict items:
    - filename  # of the input image, str
    - height    # of the input image, int
    - width     # of the input image, int
    - lines     # of the input image, list of list, N*4
    - junctions # of the input image, list of list, M*2
    '''

    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file, 'r') as _:
            self.annotations = json.load(_)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        image = io.imread(osp.join(self.root,ann['filename'])).astype(float)[:,:,:3]
        for key,_type in (['junctions',np.float32],
                          ['junctions_semantic',np.long],
                          ['edges_positive',np.long],
                          ['edges_negative',np.long],
                          ['edges_semantic',np.long],
                          ['camera_pose',np.float32]):

            ann[key] = np.array(ann[key],dtype=_type)

        for plane_type in ['planes', 'planes_negative']:
            for plane in ann[plane_type]:
                for key,_type in (['junction_idx',np.long],
                                  ['edge_idx',np.long],
                                  ['semantic',np.long],
                                  ['centroid',np.float32]):
                    plane[key] = np.array(plane[key],dtype=_type)

        for plane in ann['planes']:
            plane['parameters'] = np.array(plane['parameters'],dtype=np.float32)

        if not 'junctions_semantic' in ann:
            ann['junc_occluded'] = np.array(ann['junc_occluded'],dtype=np.bool)

        assert np.all(ann['edges_semantic'] > 0) #Assumes that the dataset also has an invalid class as 0

        if self.transform is not None:
            return self.transform(image,ann)
        return image, ann
    def image(self, idx = None, filename = None):
        if not filename:
            filename = self.annotations[idx]['filename']
        image = Image.open(osp.join(self.root,filename)).convert('RGB')
        return image
    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])
