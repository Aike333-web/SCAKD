from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def cosine_decay(initial_value, current_step, decay_steps):
    decay_factor = 0.75 * (1 + math.cos(math.pi * current_step / decay_steps))
    decayed_value = initial_value * decay_factor
    return decayed_value

def get_decay_value(initial_value, current_step, decay_steps):
    decayed_e = cosine_decay(initial_value, current_step, decay_steps / 2.0)
    decayed_m = cosine_decay(initial_value, current_step, decay_steps)
    if current_step < (decay_steps / 2.0):
        e = decayed_e
    else:
        e = 0.0
    m = decayed_m
    d = 3 - e - m
    return e,m,d

class SCAKDloss(nn.Module):
    def __init__(self, tau=0.04):
        super(SCAKDloss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.tau = tau

    def forward(self, teacher_inputs, inputs, epoch, normalized=True):
        n = inputs.size(0)
        e,m,d = get_decay_value(0.5, epoch, 240)

        if normalized:
            inputs = torch.nn.functional.normalize(inputs, dim=1)
            teacher_inputs = torch.nn.functional.normalize(teacher_inputs, dim=1)

        x1 = torch.pow(teacher_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_t = x1 + x1.t()
        dist_t.addmm_(teacher_inputs, teacher_inputs.t(), beta=1, alpha=-2)
        dist_t = dist_t.clamp(min=1e-12).sqrt()  # for numerical stability

        # Compute pairwise distance
        x1 = torch.pow(teacher_inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        x2 = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = x1 + x2.t()
        dist.addmm_(teacher_inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        temp1 = torch.diag(dist).expand(n, n).t()
        negative_index = (dist_t > temp1).float()
        negative = dist * negative_index
        negative[negative_index == 0] = 1e5
        positive_index = 1 - negative_index
        positive = dist * positive_index

        # Find hard
        dist_an = torch.min(negative, dim=1)
        dist_ap = torch.max(positive, dim=1)
        an_t = torch.gather(dist_t, 1, dist_an.indices.unsqueeze(1)).squeeze()
        ap_t = torch.gather(dist_t, 1, dist_ap.indices.unsqueeze(1)).squeeze()
        weight_an = torch.clamp_min(an_t.detach() - dist_an.values.detach(), min=0.)
        weight_ap = torch.clamp_min(dist_ap.values.detach() - ap_t.detach(), min=0.)
        weight_dist_an = weight_an * dist_an.values / self.tau
        weight_dist_ap = weight_ap * dist_ap.values / self.tau

        # Find easy
        negative[negative == 1e5] = 0.0
        positive[positive == 0] = 1e5
        dist_an_e = torch.max(negative, dim=1)
        dist_ap_e = torch.min(positive, dim=1)
        an_t_e = torch.gather(dist_t, 1, dist_an_e.indices.unsqueeze(1)).squeeze()
        ap_t_e = torch.gather(dist_t, 1, dist_ap_e.indices.unsqueeze(1)).squeeze()
        weight_an_e = torch.clamp_min(an_t_e.detach() - dist_an_e.values.detach(), min=0.)
        weight_ap_e = torch.clamp_min(dist_ap_e.values.detach() - ap_t_e.detach(), min=0.)
        weight_dist_an_e = weight_an_e * dist_an_e.values / self.tau
        weight_dist_ap_e = weight_ap_e * dist_ap_e.values / self.tau

        # Find medium
        negative[negative == 0.0] = float('nan')
        positive[positive == 1e5] = float('nan')
        median_n = torch.nanmedian(negative, dim=1)
        median_p = torch.nanmedian(positive, dim=1)
        has_nan_in_median_n = torch.isnan(median_n.values)
        has_nan_in_median_p = torch.isnan(median_p.values)

        if has_nan_in_median_n.any():
            median_n.values[has_nan_in_median_n] = 0.0
        if has_nan_in_median_p.any():
            median_p.values[has_nan_in_median_p] = 0.0

        an_t_m = torch.gather(dist_t, 1, median_n.indices.unsqueeze(1)).squeeze()
        ap_t_m = torch.gather(dist_t, 1, median_p.indices.unsqueeze(1)).squeeze()
        weight_an_m = torch.clamp_min(an_t_m.detach() - median_n.values.detach(), min=0.)
        weight_ap_m = torch.clamp_min(median_p.values.detach() - ap_t_m.detach(), min=0.)
        weight_dist_an_m = weight_an_m * median_n.values / self.tau
        weight_dist_ap_m = weight_ap_m * median_p.values / self.tau

        
        
        labels = torch.zeros(weight_dist_an.shape[0], dtype=torch.long).cuda()
        logits_e = torch.cat([weight_dist_an_e.unsqueeze(-1), weight_dist_ap_e.unsqueeze(-1)], dim=1)
        logits_m = torch.cat([weight_dist_an_m.unsqueeze(-1), weight_dist_ap_m.unsqueeze(-1)], dim=1)
        logits_d = torch.cat([weight_dist_an.unsqueeze(-1), weight_dist_ap.unsqueeze(-1)], dim=1)

        loss = self.loss(logits_e, labels) * e + self.loss(logits_m, labels) * m + self.loss(logits_d, labels) * d

        return loss

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, s_value, f_target, weight):
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()

        for i in range(num_stu):
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)
        loss = (weight * ind_loss).sum() * 0.1
        return loss

