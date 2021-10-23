# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
Modified by the authors of Video Instance Segmentation using Inter-Frame Communication Transformer.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from typing import List

from detectron2.utils.memory import retry_if_cuda_oom

from .segmentation import dice_coef
from ..util.misc import interpolate


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_dice: float = 1,
        num_classes: int = 80,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the sigmoid_focal error of the masks in the matching cost
            cost_dice: This is the relative weight of the dice loss of the masks in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_classes = num_classes
        self.num_cum_classes = [0] + np.cumsum(np.array(num_classes) + 1).tolist()

    @torch.no_grad()
    def forward(self, outputs, targets):
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].softmax(-1)
        out_mask = outputs["pred_masks"]
        B, Q, T, s_h, s_w = out_mask.shape
        t_h, t_w = targets[0]["match_masks"].shape[-2:]

        if (s_h, s_w) != (t_h, t_w):
            out_mask = out_mask.reshape(B, Q*T, s_h, s_w)
            out_mask = interpolate(out_mask, size=(t_h, t_w), mode="bilinear", align_corners=False)
            out_mask = out_mask.view(B, Q, T, t_h, t_w)

        indices = []
        for b_i in range(B):
            b_tgt_ids = targets[b_i]["labels"]
            b_out_prob = out_prob[b_i]

            cost_class = b_out_prob[:, b_tgt_ids]

            b_tgt_mask = targets[b_i]["match_masks"]
            b_out_mask = out_mask[b_i]

            # Compute the dice coefficient cost between masks
            # The 1 is a constant that doesn't change the matching as cost_class, thus omitted.
            cost_dice = retry_if_cuda_oom(dice_coef)(
                b_out_mask, b_tgt_mask
            ).to(cost_class)

            # Final cost matrix
            C = self.cost_dice * cost_dice + self.cost_class * cost_class

            indices.append(linear_sum_assignment(C.cpu(), maximize=True))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
