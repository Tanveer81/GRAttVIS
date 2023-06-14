# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_genvis_config(cfg):
    cfg.MODEL.GENVIS = CN()
    cfg.MODEL.GENVIS.LEN_CLIP_WINDOW = 5
    cfg.MODEL.GENVIS.GATED_PROP = False
    cfg.MODEL.GENVIS.USE_MEM = False
    cfg.MODEL.GENVIS.GATE_WEIGHT = 1.0