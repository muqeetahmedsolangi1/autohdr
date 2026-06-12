# AutoHDR Pipeline

A training-free HDR merge and auto-enhancement system for bracketed exposure photography — built to replicate the core output of commercial services like **AutoHDR** and **Autoenhance.ai** (real estate photo processing: HDR merge, window pull, and professional tone/color finishing).

Everything runs locally. No model training, no cloud, no API keys.

---

## How commercial "HDR" actually works (the key insight)

Services like AutoHDR/Autoenhance do **not** produce a true HDR file (radiance map + tone mapping operator). What they sell is:

1. **Exposure fusion** — per-pixel blending of bracketed shots: highlight detail from the dark exposure, shadow detail from the bright exposure.
2. **Window pull** — recovering the view through windows (sky, garden) instead of blown white.
3. **Professional finishing** — white balance, contrast curve, saturation, sharpening, learned from professional editors' work.

Their competitive moat is the **private before/after editing data** used to tune the finishing look — not a secret algorithm. The merge itself is achievable with classical, training-free techniques, which is exactly what this project does.

---

## Project structure

```
hdr-project/
├── input_brackets/        # test bracket set (memorial church, 16 exposures)
├── scripts/
│   ├── level1_fusion.py   # Level 1 — classical exposure fusion
│   ├── level2_enhance.py  # Level 2 — window pull + finishing pass
│   ├── level3_safnet.py   # Level 3 — deep learning merge (SAFNet)
│   ├── pipeline.py        # generalized pipeline (any bracket set)
│   └── app.py             # Flask web app (upload → process → download)
├── level1_output/         # Level 1 results
├── level2_output/         # Level 2 results
├── level3_output/         # Level 3 results
├── web_output/            # results produced via the web app
└── SAFNet/                # cloned SAFNet repo + pretrained weights
```

---

## Level 1 — Classical exposure fusion

**Script:** `scripts/level1_fusion.py` · **Quality: ~70-80% of commercial output**

Three steps, pure OpenCV:

1. **Alignment (`AlignMTB`)** — median threshold bitmap alignment fixes the small frame-to-frame shift in handheld brackets. Without this, the merge shows double edges (ghosting).
2. **Mertens exposure fusion (`MergeMertens`)** — for every pixel, each bracket is scored on three measures: *contrast* (local detail), *saturation* (color richness), and *well-exposedness* (distance from pure black/white). The brackets are then blended with these weights through a multi-scale Laplacian pyramid, which is why no seams or patches appear. This is the same core used by enfuse and Photomatix.
3. **8-bit output** — fusion returns float values in [0, 1]; scaled to a standard JPEG.

**Result:** all detail recovered (interior + windows), but the image looks flat — fusion is mathematically "fair", not aesthetically tuned.

## Level 2 — Window pull + finishing (the commercial look)

**Script:** `scripts/level2_enhance.py` · **Quality: matches the target look on static scenes**

Builds on Level 1 and closes the gap with two additions:

### Window pull (luminosity masking)

Mertens averages window regions, so a bright window stays milky. The fix:

- Take the **mid exposure** and build a soft mask where pixels are nearly clipped (intensity ramping 0→1 between 220 and 250).
- Feather the mask with a Gaussian blur (no hard edges).
- In masked regions, blend in the **darkest bracket** (lifted ×1.6 so the pull isn't murky).

This recovers sky/view through windows — the signature of real estate HDR.

### Finishing pass (the "editor look")

Five classical operations, in order:

| Step | Technique | Note |
|---|---|---|
| White balance | Gray-world, **30% strength** | Full-strength gray-world destroyed warm interiors (turned a golden church blue). Partial application preserves intended warmth while removing genuine casts. |
| Shadow lift | Luminance-weighted additive lift | Opens dark corners only; midtones untouched. |
| Contrast | Sine-based S-curve | Punch without crushing blacks. |
| Saturation | HSV S-channel ×1.18 | Rich but not radioactive. |
| Sharpening | Unsharp mask (σ=1.5, amount 0.6) | Final crispness. |

## Level 3 — Deep learning merge (SAFNet)

**Script:** `scripts/level3_safnet.py` · Uses [SAFNet (ECCV 2024)](https://github.com/ltkong218/SAFNet) with pretrained weights, CPU inference — no training.

The script feeds 3 brackets at 3-stop gaps (matching SAFNet's training distribution), each as a 6-channel tensor (gamma-linearized + LDR), pads to a multiple of 32, runs inference, then normalizes and tone-maps (μ-law) the predicted HDR before applying the Level 2 finishing pass.

### Honest finding — and the production recommendation

On our static test scene, **Level 3 lost to Level 2**: muddier color and flow artifacts at edges. This is expected — SAFNet is trained for *deghosting dynamic scenes* (moving people/objects between brackets) on the Kalantari dataset; a static tripod scene is out of its distribution and gains nothing from it.

**Production recipe that follows:**

> Use the classical Level 2 pipeline as the default path (fast, robust, better on static scenes). Reserve SAFNet as a fallback only when motion is detected between brackets (curtains, fans, people) — that is the one case where it beats classical alignment.

---

## Generalized pipeline + web app

### `scripts/pipeline.py`

Accepts **any** bracket set (≥2 images, any upload order):

1. Sort brackets by mean brightness (darkest first) — upload order is unknown.
2. Resize all frames to the smallest common size.
3. Align → Mertens fusion → window pull → finishing pass (same as Level 2).

### `scripts/app.py` — upload → process → download

```bash
python3 scripts/app.py
# open http://127.0.0.1:5050
```

Select 2–5 exposures of the same scene, click **Merge & Enhance**, get a preview + download link in a few seconds. Results are saved in `web_output/`.

It is also a ready-made API for integration:

```bash
curl -F "brackets=@dark.jpg" -F "brackets=@mid.jpg" -F "brackets=@bright.jpg" \
     http://127.0.0.1:5050/process
```

---

## Setup on a new machine

macOS / Linux:

```bash
git clone <repo-url> && cd autohdr
pip install -r requirements.txt        # opencv, numpy, flask, torch, transformers
python3 scripts/app.py                 # → open http://127.0.0.1:5050
```

Windows (use a virtual environment — put it on the drive with the most
free space; torch needs ~2 GB):

```powershell
git clone <repo-url> ; cd autohdr
python -m venv D:\autohdr-venv
D:\autohdr-venv\Scripts\python.exe -m pip install -r requirements.txt
D:\autohdr-venv\Scripts\python.exe scripts\app.py   # → open http://127.0.0.1:5050
```

Works on macOS, Windows, and Linux (Python 3.9+). Test brackets and demo
outputs are included in the repo, so everything runs immediately. The
SegFormer window-segmentation model (~15 MB) downloads automatically from
Hugging Face on first run; without torch/transformers installed the
pipeline still works, falling back to the classic luminosity window pull.
SAFNet code and pretrained weights are already vendored in `SAFNet/`.

## Running the levels

```bash
# requirements: python3, opencv-python, numpy, flask (+ torch for Level 3)
python3 scripts/level1_fusion.py   # → level1_output/level1_fused.jpg
python3 scripts/level2_enhance.py  # → level2_output/level2_final.jpg
python3 scripts/level3_safnet.py   # → level3_output/level3_final.jpg
python3 scripts/app.py             # → web app on port 5050
```

Test data is the classic Debevec *memorial church* bracket set (16 exposures, 1-stop apart), fetched from the `opencv_extra` repository into `input_brackets/`.

---

## Known limitations & tuning notes

- **Bracket spacing matters.** Three very widely spaced brackets (1:64:4096 ratios) produced a green-tinted fusion; five 1-stop brackets were perfect. Real camera brackets (1-2 stop gaps) are the intended input.
- **White balance is intentionally partial (30%).** Raise it for scenes with strong genuine color casts; lower it for warm scenes you want preserved.
- **Window pull threshold (220-250)** assumes 8-bit mid exposure. If windows aren't pulling, the mid bracket may already be too dark — the mask finds nothing to replace.
- **Finishing parameters are a fixed recipe** tuned by eye. The path to a per-photo adaptive look (what Autoenhance sells) is learning these parameters from before/after pairs — only needed for the last 10-15% of quality.

## Roadmap

- [ ] Test and tune on real camera real-estate brackets (phone/DSLR, 1-2 stop gaps)
- [ ] Motion detection between brackets → automatic SAFNet fallback for deghosting
- [ ] Sky enhancement / replacement for exteriors
- [ ] Cloud deployment of the Flask API (Railway/Render/EC2) for remote uploads
- [ ] Optional: learn finishing parameters from professional before/after pairs (HDRNet-style)
