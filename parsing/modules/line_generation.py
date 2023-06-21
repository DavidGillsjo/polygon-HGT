import torch
from torch import nn
import time
from parsing.utils.labels import LabelMapper

# Does not classify lines, simply generate all possible lines from junctions
class LineGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()


    def _generate_lines_from_junctions(self, juncs_pred):
        jidx = torch.combinations(torch.arange(juncs_pred.size(0), device = juncs_pred.device))
        lines_pred = juncs_pred[jidx].reshape(jidx.size(0),4)
        scores = torch.zeros([lines_pred.size(0),2], device = juncs_pred.device)
        scores[:,1] = 1
        labels = torch.ones(lines_pred.size(0), dtype=torch.long, device = juncs_pred.device)
        label_scores = torch.ones(lines_pred.size(0), device = juncs_pred.device)
        return lines_pred, labels, label_scores, jidx, scores


    # Always batch size 1, so we can do all in one take
    def forward_test(self, outputs, features, juncs_pred, annotations = None):
        device = features.device

        extra_info = {
            'time_proposal': 0.0,
            'time_matching': 0.0,
            'time_verification': 0.0,
        }


        extra_info['time_proposal'] = time.time()
        if juncs_pred.size(0) > 1:
            lines_pred, lines_label, lines_score_label, l2j_idx, scores, = self._generate_lines_from_junctions(juncs_pred)
            some_lines_valid = True
        else:
            lines_pred = None
            some_lines_valid = False
        extra_info['time_proposal'] = time.time() - extra_info['time_proposal']

        if annotations:
            width = annotations[0]['width']
            height = annotations[0]['height']
        else:
            width = images.size(3)
            height = images.size(2)

        sx = width/features.size(3)
        sy = height/features.size(2)


        output = {}

        if some_lines_valid:
            lines_pred[:,0] *= sx
            lines_pred[:,1] *= sy
            lines_pred[:,2] *= sx
            lines_pred[:,3] *= sy
            extra_info['lines_prior_ver'] = lines_pred
            extra_info['lines_prior_scoring'] = lines_pred
            extra_info['valid_junction_idx'] = torch.ones(juncs_pred.size(0), dtype=torch.bool, device=juncs_pred.device)

            output.update({
                'lines_pred': lines_pred,
                'lines_label': lines_label,
                'lines_valid_score': lines_score_label,
                'lines_label_score': lines_score_label,
                'lines_score': scores,
                'edges_pred': l2j_idx,
                'num_proposals': lines_pred.size(0),
            })
        else:
            extra_info['lines_prior_scoring'] = None
            extra_info['lines_prior_ver'] = None
            output.update({
                'lines_pred': torch.tensor([]),
                'num_proposals': 0
                })

        return output, extra_info, None

    # Handle batch losses, store state variables for forward_train_image
    def forward_train_batch(self, outputs, features, targets):
        pass

    def reset_and_get_loss_dict(self):
        return {}

    # Handle line detection, called for every image.
    # img_idx is the index for the image in the batch, so we can correctly use our state variables
    def forward_train_image(self, img_idx, juncs_pred, meta):
        device = juncs_pred.device

        # No junctions, just add static training examples
        if juncs_pred.size(0) < 2:
            return None

        lines_pred, lines_label, lines_label_score, l2j_idx, scores, = self._generate_lines_from_junctions(juncs_pred)

        output = {
            'edges_pred': l2j_idx,
            'lines_pred': lines_pred,
            'lines_logits': scores
        }
        return output
