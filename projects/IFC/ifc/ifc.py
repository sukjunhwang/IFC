# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances

from .models.backbone import Joiner
from .models.ifc import IFC, SetCriterion
from .models.matcher import HungarianMatcher
from .models.position_encoding import PositionEmbeddingSine
from .models.segmentation import MaskHead, segmentation_postprocess
from .models.transformer import IFCTransformer
from .structures.clip_output import Videos, Clips
from .util.misc import NestedTensor, _max_by_axis, interpolate

__all__ = ["Ifc"]


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class Ifc(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.clip_stride = cfg.MODEL.IFC.CLIP_STRIDE
        self.merge_on_cpu = cfg.MODEL.IFC.MERGE_ON_CPU
        self.is_multi_cls = cfg.MODEL.IFC.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.IFC.APPLY_CLS_THRES

        self.is_coco = cfg.DATASETS.TEST[0].startswith("coco")
        self.num_classes = cfg.MODEL.IFC.NUM_CLASSES
        self.mask_stride = cfg.MODEL.IFC.MASK_STRIDE
        self.match_stride = cfg.MODEL.IFC.MATCH_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.IFC.HIDDEN_DIM
        num_queries = cfg.MODEL.IFC.NUM_OBJECT_QUERIES

        # Transformer parameters:
        nheads = cfg.MODEL.IFC.NHEADS
        dropout = cfg.MODEL.IFC.DROPOUT
        dim_feedforward = cfg.MODEL.IFC.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.IFC.ENC_LAYERS
        dec_layers = cfg.MODEL.IFC.DEC_LAYERS
        pre_norm = cfg.MODEL.IFC.PRE_NORM
        num_memory_bus = cfg.MODEL.IFC.NUM_MEMORY_BUS

        # Loss parameters:
        mask_weight = cfg.MODEL.IFC.MASK_WEIGHT
        dice_weight = cfg.MODEL.IFC.DICE_WEIGHT
        deep_supervision = cfg.MODEL.IFC.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.IFC.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = IFCTransformer(
            num_frames=self.num_frames,
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            num_memory_bus=num_memory_bus,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        mask_head = MaskHead(hidden_dim, [1024, 512], self.num_frames)

        self.detr = IFC(
            backbone, transformer, mask_head,
            num_classes=self.num_classes, num_queries=num_queries,
            num_frames=self.num_frames, aux_loss=deep_supervision
        )
        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_dice=dice_weight,
            num_classes=self.num_classes,
        )
        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks", "cardinality"]
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
            num_frames=self.num_frames
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.merge_device = "cpu" if self.merge_on_cpu else self.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training or self.is_coco:
            images = self.preprocess_image(batched_inputs)
            output = self.detr(images)

            if self.training:
                gt_instances = []
                for video in batched_inputs:
                    for frame in video["instances"]:
                        gt_instances.append(frame.to(self.device))

                targets = self.prepare_targets(batched_inputs)
                loss_dict = self.criterion(output, targets)
                weight_dict = self.criterion.weight_dict
                for k in loss_dict.keys():
                    if k in weight_dict:
                        loss_dict[k] *= weight_dict[k]
                return loss_dict
            else:
                return self.inference_image(output, batched_inputs, images)

        # Youtube-VIS evaluation should be treated in a different manner.
        else:
            # TODO provide sequential runner for long videos.
            # NOTE we assume that only a single video is taken as an input.
            video = self.preprocess_image(batched_inputs)

            backbone_tensor, backbone_pos = self.detr.forward_pre_backbone(video)

            video_length = len(video)
            image_size = video.tensor.shape[-2:]
            interim_size = (math.ceil(image_size[0] / 8), math.ceil(image_size[1] / 8))

            video_output = Videos(
                self.num_frames, video_length, self.num_classes, interim_size, self.merge_device
            )

            is_last_clip = False
            for start_idx in range(0, video_length, self.clip_stride):
                end_idx = start_idx + self.num_frames
                if end_idx >= video_length:
                    is_last_clip = True
                    start_idx, end_idx = max(0, video_length - self.num_frames), video_length

                frame_idx = list(range(start_idx, end_idx))

                clip_backbone_tensor = [t[frame_idx] for t in backbone_tensor]
                clip_backbone_pos = [p[frame_idx] for p in backbone_pos]

                output = self.detr.forward_post_backbone(clip_backbone_tensor, clip_backbone_pos, is_train=False)

                _clip_results = self.inference_clip(output, image_size)
                clip_results = Clips(frame_idx, _clip_results.to(self.merge_device))

                video_output.update(clip_results)

                if is_last_clip:
                    break

            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            pred_cls, pred_masks = video_output.get_result((height, width)) # NxHxW / NxC

            return self.inference_video(pred_cls, pred_masks, (height, width))

    def prepare_targets(self, targets):
        gt_instances = []

        _sizes = []
        for targets_per_video in targets:
            _sizes += [list(t.image_size) for t in targets_per_video["instances"]]
        max_size = _max_by_axis(_sizes)

        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames] + max_size
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            o_h, o_w = gt_masks_per_video.shape[-2:]
            l_h, l_w = math.ceil(o_h/self.mask_stride), math.ceil(o_w/self.mask_stride)
            m_h, m_w = math.ceil(o_h/self.match_stride), math.ceil(o_w/self.match_stride)

            gt_masks_for_loss  = interpolate(gt_masks_per_video, size=(l_h, l_w), mode="bilinear", align_corners=False)
            gt_masks_for_match = interpolate(gt_masks_per_video, size=(m_h, m_w), mode="bilinear", align_corners=False)
            gt_instances[-1].update({"masks": gt_masks_for_loss, "match_masks": gt_masks_for_match})

        return gt_instances

    def inference_clip(self, output, image_size):
        mask_cls = output["pred_logits"][0]
        mask_pred = output["pred_masks"][0]

        # For each mask we assign the best class or the second best if the best on is `no_object`.
        _idx = self.num_classes + 1
        mask_cls = F.softmax(mask_cls, dim=-1)[:, :_idx]
        scores, labels = mask_cls.max(-1)

        valid = (labels < self.num_classes)
        scores = scores[valid]
        labels = labels[valid]
        mask_cls = mask_cls[valid]
        mask_pred = mask_pred[valid]

        results = Instances(image_size)
        results.scores = scores
        results.pred_classes = labels
        results.cls_probs = mask_cls
        results.pred_masks = mask_pred

        return results

    def inference_video(self, pred_cls, pred_masks, image_size):
        if len(pred_cls) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(pred_cls[:, :-1] > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]
                labels = is_above_thres[1]
                pred_masks = pred_masks[is_above_thres[0]]
            else:
                scores, labels = pred_cls[:, :-1].max(-1)

            masks = pred_masks > 0.5

            out_scores = scores.tolist()
            out_labels = labels.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": image_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    def inference_image(self, output, batched_inputs, images):
        mask_cls = output["pred_logits"]
        mask_pred = output["pred_masks"]

        results = self._inference_image(
            mask_cls, mask_pred, images.image_sizes
        )
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = segmentation_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _inference_image(self, mask_cls, mask_pred, image_sizes):
        """
        Arguments:
            mask_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(mask_cls) == len(image_sizes) // self.num_frames
        results = []

        max_size = _max_by_axis([list(s) for s in image_sizes])
        b, n = mask_pred.shape[:2]
        mask_pred = interpolate(
            mask_pred.flatten(0,1), size=max_size, mode='bilinear', align_corners=False
        ) > 0.0
        mask_pred = mask_pred.view(b, n, *mask_pred.shape[-3:])

        # For each mask we assign the best class or the second best if the best on is `no_object`.
        _idx = self.num_classes + 1
        scores, labels = F.softmax(mask_cls, dim=-1)[:, :, :_idx].max(-1)

        for i, (scores_per_clip, labels_per_clip, mask_pred_per_clip) in enumerate(zip(
            scores, labels, mask_pred
        )):
            valid = (labels_per_clip < self.num_classes)
            scores_per_clip = scores_per_clip[valid]
            labels_per_clip = labels_per_clip[valid]
            mask_pred_per_clip = mask_pred_per_clip[valid]

            for j in range(self.num_frames):
                image_size = image_sizes[i*self.num_frames + j]
                result = Instances(image_size)

                pred_masks = mask_pred_per_clip[:, j, :image_size[0], :image_size[1]]

                pred_boxes = torch.zeros(pred_masks.shape[0], 4, dtype=torch.float32)
                x_any = torch.any(pred_masks, dim=1)
                y_any = torch.any(pred_masks, dim=2)
                for idx in range(pred_masks.shape[0]):
                    x = torch.where(x_any[idx, :])[0]
                    y = torch.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        pred_boxes[idx, :] = torch.as_tensor(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                        )
                pred_boxes = Boxes(pred_boxes)

                result.scores = scores_per_clip
                result.pred_classes = labels_per_clip
                result.pred_masks = pred_masks
                result.pred_boxes = pred_boxes
                results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images)
        return images
