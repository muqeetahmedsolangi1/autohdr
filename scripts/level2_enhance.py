"""Level 2 — Window pull + professional tone/color finishing.

Builds on Level 1 fusion, then:
 1. Window pull: where the mid exposure is clipped (blown highlights),
    feather-blend detail from the darkest bracket.
 2. Finishing pass: gray-world white balance, S-curve contrast,
    shadow lift, saturation boost, unsharp-mask sharpening.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input_brackets"
OUTPUT = ROOT / "level2_output"

BRACKETS = [
    "memorial00.png",  # darkest
    "memorial03.png",
    "memorial06.png",  # mid
    "memorial09.png",
    "memorial12.png",  # brightest
]
DARK_IDX = 0
MID_IDX = 2


def load_brackets():
    images = []
    for name in BRACKETS:
        img = cv2.imread(str(INPUT / name))
        if img is None:
            sys.exit(f"missing input: {name}")
        images.append(img)
    return images


def fuse(images):
    cv2.createAlignMTB().process(images, images)
    return cv2.createMergeMertens().process(images)  # float32 ~0-1


def window_pull(fused, images):
    """Blend dark-bracket detail wherever the mid exposure is clipped."""
    mid_gray = cv2.cvtColor(images[MID_IDX], cv2.COLOR_BGR2GRAY)
    # 0 below 220, ramps to 1 at 250 — soft clipping mask
    mask = np.clip((mid_gray.astype(np.float32) - 220.0) / 30.0, 0, 1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]

    dark = images[DARK_IDX].astype(np.float32) / 255.0
    # brighten the dark bracket a touch so the pull isn't murky
    dark = np.clip(dark * 1.6, 0, 1)
    return fused * (1 - mask) + dark * mask


def gray_world_wb(img, strength=0.3):
    # full gray-world kills legitimately warm interiors — apply partially
    means = img.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-6)
    gains = 1 + (gains - 1) * strength
    return np.clip(img * gains, 0, 1)


def s_curve(img, strength=0.12):
    return np.clip(img + strength * np.sin(2 * np.pi * (img - 0.5)) * -1, 0, 1)


def lift_shadows(img, amount=0.08):
    luma = img.mean(axis=2, keepdims=True)
    lift = amount * np.clip(1 - luma * 3, 0, 1)
    return np.clip(img + lift, 0, 1)


def boost_saturation(img, factor=1.18):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out.astype(np.float32) / 255.0


def sharpen(img, amount=0.6, sigma=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + amount * (img - blur), 0, 1)


def main():
    OUTPUT.mkdir(exist_ok=True)
    images = load_brackets()
    fused = fuse(images)

    pulled = window_pull(fused, images)
    cv2.imwrite(
        str(OUTPUT / "level2_window_pull_only.jpg"),
        (pulled * 255).clip(0, 255).astype("uint8"),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    img = gray_world_wb(pulled)
    img = lift_shadows(img)
    img = s_curve(img)
    img = boost_saturation(img)
    img = sharpen(img)

    out = OUTPUT / "level2_final.jpg"
    cv2.imwrite(
        str(out),
        (img * 255).clip(0, 255).astype("uint8"),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
