"""Level 1 — Classical HDR via exposure fusion.

Align bracketed exposures (AlignMTB) then merge with Mertens
exposure fusion. This is the same core used by enfuse/Photomatix.
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "input_brackets"
OUTPUT = ROOT / "level1_output"

BRACKETS = [
    "memorial00.png",  # darkest — window/highlight detail
    "memorial03.png",
    "memorial06.png",  # mid
    "memorial09.png",
    "memorial12.png",  # brightest — shadow detail
]


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
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0,
    )
    return merge.process(images)  # float32, ~0-1


def main():
    OUTPUT.mkdir(exist_ok=True)
    images = load_brackets()
    fused = fuse(images)
    result = (fused * 255).clip(0, 255).astype("uint8")
    out = OUTPUT / "level1_fused.jpg"
    cv2.imwrite(str(out), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"saved {out} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
