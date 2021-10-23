# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_ifc_config(cfg):
    """
    Add config for IFC.
    """
    cfg.MODEL.IFC = CN()
    cfg.MODEL.IFC.NUM_CLASSES = 80

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 5
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # LOSS
    cfg.MODEL.IFC.MASK_WEIGHT = 3.0
    cfg.MODEL.IFC.DICE_WEIGHT = 3.0
    cfg.MODEL.IFC.DEEP_SUPERVISION = True
    cfg.MODEL.IFC.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.IFC.MASK_STRIDE = 4
    cfg.MODEL.IFC.MATCH_STRIDE = 4

    # TRANSFORMER
    cfg.MODEL.IFC.NHEADS = 8
    cfg.MODEL.IFC.DROPOUT = 0.1
    cfg.MODEL.IFC.DIM_FEEDFORWARD = 2048
    cfg.MODEL.IFC.ENC_LAYERS = 3
    cfg.MODEL.IFC.DEC_LAYERS = 3
    cfg.MODEL.IFC.PRE_NORM = False
    cfg.MODEL.IFC.NUM_MEMORY_BUS = 8

    cfg.MODEL.IFC.HIDDEN_DIM = 256
    cfg.MODEL.IFC.NUM_OBJECT_QUERIES = 100

    # Evaluation
    cfg.MODEL.IFC.CLIP_STRIDE = 1
    cfg.MODEL.IFC.MERGE_ON_CPU = False
    cfg.MODEL.IFC.MULTI_CLS_ON = True
    cfg.MODEL.IFC.APPLY_CLS_THRES = 0.01

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
