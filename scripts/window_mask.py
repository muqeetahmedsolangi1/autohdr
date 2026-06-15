"""Semantic masks via SegFormer (ADE20K-pretrained).

Provides soft masks for windows (window pull), walls+ceiling (forced
white-balance) and floors (protected from neutralization).

The model (~15MB) downloads automatically from Hugging Face on first
use and runs on CPU in ~1-2s. If torch/transformers are not installed,
the getters return None and the pipeline falls back to classical
luminosity logic.
"""
import cv2
import numpy as np

# ADE20K class ids
WINDOW_CLASSES = (8, 14)        # windowpane, door (glass doors)
WALLCEIL_CLASSES = (0, 5)       # wall, ceiling
FLOOR_CLASSES = (3, 28)         # floor, rug
LAMP_CLASSES = (36, 82, 85)     # lamp, light/sconce, chandelier
CABINET_CLASSES = (10, 15, 24)  # cabinet, table, shelf (warm wood surfaces)

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _processor


def get_label_masks(bgr):
    """Dict of soft float32 masks {'window','wallceil','floor','lamp',
    'cabinet'} at full resolution, or None if segmentation is
    unavailable. Individual entries are None when that class is absent."""
    try:
        import torch
        model, processor = _load()
    except Exception:
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # 1 x classes x h/4 x w/4
    labels = logits.argmax(dim=1)[0].numpy().astype(np.uint8)

    h, w = bgr.shape[:2]

    def soft(classes):
        m = np.isin(labels, classes).astype(np.float32)
        if m.sum() < 10:
            return None
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(m, 0, 1)

    return {
        "window": soft(WINDOW_CLASSES),
        "wallceil": soft(WALLCEIL_CLASSES),
        "floor": soft(FLOOR_CLASSES),
        "lamp": soft(LAMP_CLASSES),
        "cabinet": soft(CABINET_CLASSES),
    }


def get_window_mask(bgr, feather=51):
    """Back-compat: soft window mask only."""
    masks = get_label_masks(bgr)
    return None if masks is None else masks["window"]
