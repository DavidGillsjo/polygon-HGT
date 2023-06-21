import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

class PlaneEncoder(object):
    def __init__(self, cfg):
        pass
    def __call__(self,annotations):
        targets = []
        for ann in annotations:
            t = self._process_per_image(ann)
            targets.append(t)

        return default_collate(targets)

    def _process_per_image(self,ann):
        junctions = ann['junctions']
        device = junctions.device
        centroids = torch.zeros([len(ann['planes']), 2], device=device)
        plane_sematics = torch.zeros([len(ann['planes'])], dtype=int, device=device)
        for i, p in enumerate(ann['planes']):
            centroids[i] = p['centroid']
            plane_sematics[i] = int(p['semantic'])

        height, width = ann['height'], ann['width']
        # Mask for where we have centroid
        pc_mask = torch.zeros((height,width), dtype=torch.float16, device=device)
        # Offset for each centroid
        pc_off = torch.zeros((2,height,width),dtype=torch.float32, device=device)
        # Label for each centroid
        pc_label = torch.zeros((height,width), dtype=torch.long, device=device)

        xint,yint = centroids[:,0].long(), centroids[:,1].long()
        off_x = centroids[:,0] - xint.float()-0.5
        off_y = centroids[:,1] - yint.float()-0.5
        pc_off[0,yint,xint] = off_x
        pc_off[1,yint,xint] = off_y
        pc_mask[yint,xint] = 1


        pc_label[yint, xint] = plane_sematics
        target = {'pc_mask':pc_mask[None],
                'pc_label':pc_label[None],
                'pc_off':pc_off
               }

        return target
