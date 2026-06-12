"""Level 3 — Deep learning HDR merge with pretrained SAFNet (ECCV 2024).

Uses 3 brackets at 3-stop gaps (matching SAFNet's training distribution),
runs inference on CPU, then tone maps the linear HDR output with mu-law
and applies the Level 2 finishing pass.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "SAFNet"))
from models.SAFNet import SAFNet  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import level2_enhance as l2  # noqa: E402

INPUT = ROOT / "input_brackets"
OUTPUT = ROOT / "level3_output"

# short / mid / long at 1:8:64 exposure ratios (3-stop gaps, like Kalantari)
BRACKETS = [("memorial03.png", 1.0), ("memorial06.png", 8.0), ("memorial09.png", 64.0)]


def load_inputs():
    images = []
    for name, _ in BRACKETS:
        img = cv2.imread(str(INPUT / name))
        if img is None:
            sys.exit(f"missing input: {name}")
        images.append(img)
    cv2.createAlignMTB().process(images, images)

    tensors = []
    for img, (_, expo) in zip(images, BRACKETS):
        ldr = (img[:, :, ::-1] / 255.0).astype(np.float32)  # RGB, 0-1
        lin = (ldr ** 2.2) / expo
        six = np.concatenate([lin, ldr], axis=2)
        tensors.append(torch.from_numpy(six).permute(2, 0, 1).unsqueeze(0))
    return tensors


def pad_to(t, mult=32):
    _, _, h, w = t.shape
    ph, pw = (mult - h % mult) % mult, (mult - w % mult) % mult
    return torch.nn.functional.pad(t, (0, pw, 0, ph), mode="replicate"), h, w


def mu_law(hdr, mu=5000.0):
    return np.log(1 + mu * hdr) / np.log(1 + mu)


def main():
    OUTPUT.mkdir(exist_ok=True)
    model = SAFNet().eval()
    state = torch.load(
        ROOT / "SAFNet/checkpoints/SAFNet_siggraph17.pth", map_location="cpu"
    )
    model.load_state_dict(state)

    img0, img1, img2 = load_inputs()
    img0, h, w = pad_to(img0)
    img1, _, _ = pad_to(img1)
    img2, _, _ = pad_to(img2)

    with torch.no_grad():
        _, hdr = model(img0, img1, img2)

    hdr = hdr[0, :, :h, :w].permute(1, 2, 0).numpy().clip(0, None)  # RGB linear
    hdr = hdr / max(np.percentile(hdr, 99.5), 1e-6)  # normalize before tone map

    tone = mu_law(hdr).clip(0, 1)[:, :, ::-1]  # to BGR for cv2/level2 helpers
    cv2.imwrite(
        str(OUTPUT / "level3_safnet_mulaw.jpg"),
        (tone * 255).clip(0, 255).astype("uint8"),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    img = l2.gray_world_wb(tone.astype(np.float32))
    img = l2.lift_shadows(img)
    img = l2.s_curve(img)
    img = l2.boost_saturation(img)
    img = l2.sharpen(img)
    out = OUTPUT / "level3_final.jpg"
    cv2.imwrite(
        str(out),
        (img * 255).clip(0, 255).astype("uint8"),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
