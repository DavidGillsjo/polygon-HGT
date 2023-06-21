import torch

def scaled_sigmoid(x, offset = 0.0, scale=1.0):
    return scale*(torch.sigmoid(x) + offset)

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None, scale=1.0):
    logp = scaled_sigmoid(logits, offset=offset, scale=scale)
    loss = torch.abs(logp-targets)

    if mask is not None:
        w = mask.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(mask/w)

    return loss.mean()
