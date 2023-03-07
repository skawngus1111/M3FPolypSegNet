import torch

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, jaccard_score)

def calculate_top1_error(output, target) :
    _, rank1 = torch.max(output, 1)
    correct_top1 = (rank1 == target).sum().item()

    return correct_top1

def calculate_top5_error(output, target) :
    _, top5 = output.topk(5, 1, True, True)
    top5 = top5.t()
    correct5 = top5.eq(target.view(1, -1).expand_as(top5))

    for k in range(6):
        correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

    correct_top5 = correct_k.item()

    return correct_top5

def metrics(true, pred) :
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.int_)

    true = np.asarray(true.flatten(), dtype=np.int64)
    pred = np.asarray(pred.flatten(), dtype=np.int64)

    acc = accuracy_score(true, pred)
    pre = precision_score(true, pred, average='macro')
    rec = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    iou = jaccard_score(true, pred, average='macro')

    return acc, f1, pre, rec, iou