import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics.pairwise import pairwise_distances


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''

        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy','gce']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels, weights=None, reduction=True):
        if self.loss_type == 'crossentropy':
            wf = F.log_softmax(wf, dim=-1)
            loss = F.nll_loss(wf, labels, weights)
            return loss
        elif self.loss_type == 'gce':
            q = 0.1
            Yg_soft = torch.gather(torch.softmax(wf, dim=1), 1, torch.unsqueeze(labels, 1))
            loss = ((1 - (Yg_soft ** q)) / q)
            return torch.mean(loss)
        else:
            ys = self.s
            if self.loss_type == 'cosface':
                numerator = ys * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))
            Yg = torch.gather(wf, 1, torch.unsqueeze(labels, 1)).squeeze()
            denominator = torch.exp(numerator) + torch.sum(torch.exp(ys * wf), dim=1) - torch.exp(ys * Yg)
            loss = numerator - torch.log(denominator)
            if reduction:
                return -torch.mean(loss)
            else:
                return -loss

