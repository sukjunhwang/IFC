# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from detectron2.structures import Instances
from detectron2.layers.blocks import DepthwiseSeparableConv2d

from ..util.misc import interpolate
from .misc import MLP


class MaskHead(nn.Module):
    def __init__(self, hidden_dim, fpn_dims, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.lay1 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(32, hidden_dim)
        self.lay2 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(32, hidden_dim)
        self.lay3 = torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(32, hidden_dim)
        self.out_lay = DepthwiseSeparableConv2d(hidden_dim, hidden_dim, 5, padding=2, activation1=F.relu, activation2=F.relu)

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], hidden_dim, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], hidden_dim, 1)

        self.convert_to_weight = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: List[Tensor], tq: Tensor):
        x = self.lay1(x)
        x = self.gn1(x)

        cur_fpn = self.adapter1(fpns[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay2(x)
        x = self.gn2(x)

        cur_fpn = self.adapter2(fpns[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        BT, C, H, W = x.shape
        L, B, N, C = tq.shape
        T = BT // B

        x = self.out_lay(x)
        w = self.convert_to_weight(tq).permute(1,0,2,3)
        w = w.unsqueeze(1).repeat(1,T,1,1,1)

        mask_logits = F.conv2d(x.view(1, BT*C, H, W), w.reshape(B*T*L*N, C, 1, 1), groups=BT)
        mask_logits = mask_logits.view(B, T, L, N, H, W).permute(2, 0, 3, 1, 4, 5)

        return mask_logits


def dice_coef(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1).unsqueeze(1)
    targets = targets.flatten(1).unsqueeze(0)
    numerator = 2 * (inputs * targets).sum(2)
    denominator = inputs.sum(-1) + targets.sum(-1)

    # NOTE coef doesn't be subtracted to 1 as it is not necessary for computing costs
    coef = (numerator + 1) / (denominator + 1)
    return coef


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_coef(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1)
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1)

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    coef = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        coef = alpha_t * coef

    return coef.mean(2)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def segmentation_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    For instance segmentation whose masks are size of batched output,
    not regional sizes as from R-CNN based predictor.
    """
    scale_x, scale_y = (float(output_width) / results.image_size[1], float(output_height) / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    if results.has("pred_masks"):
        results.pred_masks = interpolate(
            results.pred_masks.float().unsqueeze(1), size=(output_height, output_width),
            mode='bilinear'
        ).squeeze(1) > 0.5

    results = results[output_boxes.nonempty()]

    return results
