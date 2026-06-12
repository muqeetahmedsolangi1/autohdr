"""Generalized AutoHDR pipeline for arbitrary bracket sets.

Takes any number (>=2) of bracketed exposures in any order:
sorts by brightness, aligns (ECC homography), fuses (Mertens),
pulls windows from the best-exposed bracket (SegFormer mask, with
luminosity fallback), then applies the finishing pass:
white balance -> mixed-light neutralization -> levels -> exposure
-> S-curve -> vibrance -> two-stage sharpening.
"""
import cv2
import numpy as np

import window_mask


# ---------------------------------------------------------------- alignment

def align_ecc(images, work_width=1000):
    """Align all brackets to the middle one with a full homography.

    ECC is estimated on a downscaled, histogram-equalized copy (robust to
    exposure differences, fast), then the homography is rescaled and
    applied at full resolution. Falls back to the unwarped image when a
    bracket fails to converge.
    """
    ref = len(images) // 2
    h, w = images[0].shape[:2]
    scale = min(1.0, work_width / w)
    small = [
        cv2.equalizeHist(
            cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None,
                       fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )
        for im in images
    ]
    S = np.diag([scale, scale, 1.0])
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

    aligned = []
    for i, im in enumerate(images):
        if i == ref:
            aligned.append(im)
            continue
        warp = np.eye(3, dtype=np.float32)
        try:
            cv2.findTransformECC(small[ref], small[i], warp,
                                 cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
            H = (np.linalg.inv(S) @ warp.astype(np.float64) @ S).astype(np.float32)
            aligned.append(cv2.warpPerspective(
                im, H, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE))
        except cv2.error:
            aligned.append(im)
    return aligned


# -------------------------------------------------------------- window pull

def build_window_mask(mid, seg_mask=None):
    """Soft mask of blown window glass.

    Gated by actual clipping in the mid exposure (>= 235) so a wall or
    frame can never be touched; segmentation, when available, focuses
    the pull on real windows (full strength) while elsewhere-clipped
    pixels still get a gentle pull. A wrong segmentation label cannot
    create an artifact because non-clipped pixels stay at zero.
    """
    mid_gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ramp = np.clip((mid_gray - 235.0) / 20.0, 0, 1)
    if seg_mask is not None:
        ramp = ramp * (0.35 + 0.65 * seg_mask)
    k = max(31, int(mid_gray.shape[1] * 0.008) | 1)
    return cv2.GaussianBlur(ramp, (k, k), 0)


def window_content(dark_bracket, wmask, target=0.72):
    """Prepare the view-through-the-glass layer from the darkest bracket.

    This is the Autoenhance recipe: take the bracket with the most
    window information, enhance it separately (gain to a bright-but-
    detailed level, neutral WB, gentle vibrance), and blend it back as
    the LAST tonal step — so the interior brightening can never wash
    the windows back out.
    """
    sel = wmask > 0.5
    if not sel.any():
        return None
    dark = dark_bracket.astype(np.float32) / 255.0
    med = float(np.median(dark.mean(axis=2)[sel]))
    gain = np.clip(target / max(med, 1e-3), 1.0, 2.4)
    content = np.clip(dark * gain, 0, 1)
    content = white_patch_wb(content, percentile=85.0, strength=1.0)
    return vibrance(content, factor=1.15)


def whiten_lamps(img, strength=1.0, max_area_frac=0.015):
    """Turn warm light sources white (the AutoHDR rendering).

    Finds small, bright, warm blobs — bulbs, lamp shades, recessed-can
    glow pools — and pulls their color to neutral in LAB. Large warm
    areas (wood floors, furniture) are excluded by the component-size
    filter, so only light sources are affected.
    """
    lab = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    b = lab[..., 2] - 128.0
    warm = ((L > 0.65) & (b > 8)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(warm)
    if n < 2:
        return img
    h, w = warm.shape
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < h * w * max_area_frac
    keep = small[labels].astype(np.float32)
    keep = cv2.GaussianBlur(keep, (0, 0), max(4.0, w * 0.002)) * strength
    lab[..., 1] = (lab[..., 1] - 128.0) * (1 - keep) + 128.0
    lab[..., 2] = (lab[..., 2] - 128.0) * (1 - keep) + 128.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------- finishing

def white_patch_wb(img, percentile=92.0, strength=1.0, exclude=None):
    """Neutralize color cast using the brightest non-clipped pixels.

    Real estate interiors almost always have white walls/ceilings, so the
    bright regions are a reliable neutral reference — unlike gray-world,
    which is skewed by wood floors and furniture.

    `exclude` (window mask) is essential: without it the measurement is
    dominated by window glass showing bluish daylight, and "correcting"
    that blue to neutral turns the whole interior golden.
    """
    luma = img.mean(axis=2)
    valid = luma < 0.98
    if exclude is not None:
        valid &= exclude < 0.5
    vals = luma[valid]
    if vals.size < 1000:
        return img
    ref = img[valid & (luma >= np.percentile(vals, percentile))]
    if len(ref) < 100:  # everything clipped — nothing to measure
        return img
    means = ref.mean(axis=0)
    gains = means.max() / np.maximum(means, 1e-6)
    gains = 1 + (gains - 1) * strength
    return np.clip(img * gains, 0, 1)


def neutralize_whites(img, strength=1.0, boost=None, protect=None):
    """Remove local warm/cool casts from surfaces that should be white.

    Mixed lighting (tungsten bulbs + daylight) tints walls and ceilings
    differently across the frame; one global WB can't fix both. Pulled
    toward neutral in LAB:
      * bright, low-chroma surfaces (whites with a mild cast);
      * `boost` mask (segmented walls+ceiling): forced neutral even when
        warm light tinted them heavily — color alone can't separate a
        warm-lit white wall from a wood floor, semantics can;
      * very bright pixels regardless of chroma — lamps and bulbs, which
        the AutoHDR look renders white rather than orange.
    `protect` (segmented floor/rug) is exempted so floors keep their color.
    """
    lab = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    w_walls = (np.clip((L - 0.70) / 0.12, 0, 1)
               * np.clip((22.0 - chroma) / 8.0, 0, 1))
    if boost is not None:
        w_seg = (boost * np.clip((L - 0.55) / 0.20, 0, 1)
                 * np.clip((35.0 - chroma) / 10.0, 0, 1))
        w_walls = np.maximum(w_walls, w_seg)
    if protect is not None:
        w_walls = w_walls * (1 - protect)
    w_lamps = np.clip((L - 0.85) / 0.10, 0, 1)
    w = cv2.GaussianBlur(np.maximum(w_walls, w_lamps), (0, 0), 8) * strength
    lab[..., 1] = a * (1 - w) + 128.0
    lab[..., 2] = b * (1 - w) + 128.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.astype(np.float32) / 255.0


def auto_levels(img, black_pct=0.4, white_pct=99.7, max_black=0.25,
                exclude=None):
    """Anchor the black and white points (Lightroom-style auto levels).

    Brightening alone leaves the image hazy: shadows that should be black
    turn gray. Stretching so the darkest percentile maps to 0 and the
    brightest to 1 restores contrast and makes white walls truly white.
    The same stretch is applied to all channels so colors don't shift.

    `exclude` (window mask): window pixels are left out of the percentile
    measurement so the white point anchors on the WALLS — otherwise the
    windows are always the brightest pixels and the walls stay gray.
    """
    luma = img.mean(axis=2)
    sample = luma if exclude is None else luma[exclude < 0.5]
    if sample.size < 1000:
        sample = luma
    lo = min(float(np.percentile(sample, black_pct)), max_black)
    hi = float(np.percentile(sample, white_pct))
    if hi - lo < 0.1:
        return img
    # measured from AutoHDR outputs: blacks sit at ~17/255 (soft, not
    # crushed) and whites at ~235/255 (bright but never blasted); the
    # later gamma brighten lifts the floor further, hence 0.02 here
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return out * (0.91 - 0.02) + 0.02


def auto_exposure(img, target=0.60):
    """Brighten until the median luminance hits the target (gamma-based).

    Gamma lifts midtones without clipping the highlights we just
    recovered in the window pull. Only ever brightens, never darkens.
    """
    luma = img.mean(axis=2)
    med = max(float(np.median(luma)), 1e-3)
    if med >= target:
        return img
    gamma = np.clip(np.log(target) / np.log(med), 0.45, 1.0)
    return np.clip(img, 0, 1) ** gamma


def match_neutral_whites(img, strength=0.85):
    """Global temperature/tint trim: shift LAB a/b so the image's bright
    pixels land on neutral (a=b=128). This is the measured difference
    between our output and AutoHDR's — their whites sit at exactly
    128/128 while warm casts leave ours a few points yellow."""
    lab = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    bright = lab[..., 0] > 0.78 * 255
    if bright.sum() < 1000:
        return img
    lab[..., 1] += (128.0 - float(lab[..., 1][bright].mean())) * strength
    lab[..., 2] += (128.0 - float(lab[..., 2][bright].mean())) * strength
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out.astype(np.float32) / 255.0


def s_curve(img, strength=0.08):
    """True S-curve: deepens shadows, lifts highlights.

    Note: level2_enhance.s_curve has the sign inverted (it flattens
    contrast — discovered comparing output dullness against AutoHDR
    references), so the pipeline uses this corrected version.
    """
    return np.clip(img + strength * np.sin(2 * np.pi * (img - 0.5)), 0, 1)


def vibrance(img, factor=1.25):
    """Boost muted colors more than already-saturated ones.

    Plain saturation pushes colorful pixels (plants, flowers) into neon;
    vibrance scales the boost by (1 - S) so they barely change while
    wood floors and fabrics gain richness.
    """
    hsv = cv2.cvtColor((img * 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + (factor - 1) * (1 - s)), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out.astype(np.float32) / 255.0


def sharpen_two_stage(img, fine_amount=0.9, clarity_amount=0.2):
    """Fine unsharp mask for edge crispness + wide-radius local contrast
    (the Lightroom 'Clarity' effect). Runs last in the pipeline."""
    fine = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = np.clip(img + fine_amount * (img - fine), 0, 1)
    wide = cv2.GaussianBlur(img, (0, 0), 15)
    return np.clip(img + clarity_amount * (img - wide), 0, 1)


# ------------------------------------------------------------------ pipeline

MAX_WIDTH = int(__import__("os").environ.get("AUTOHDR_MAX_WIDTH", 4000))


def process_brackets(images, enhance=True, pull_windows=False):
    if len(images) < 2:
        raise ValueError("need at least 2 bracketed exposures")

    # darkest first — upload order is unknown
    images = sorted(images, key=lambda im: im.mean())

    # alignment needs equal sizes; resize to the smallest frame, capped at
    # MAX_WIDTH — 45MP x 7 brackets needs more RAM than a laptop has, and
    # 4000px output is plenty for listings (raise AUTOHDR_MAX_WIDTH on a
    # bigger machine)
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    if w > MAX_WIDTH:
        h = int(h * MAX_WIDTH / w)
        w = MAX_WIDTH
    images = [
        im if im.shape[:2] == (h, w)
        else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        for im in images
    ]

    images = align_ecc(images)
    fused = cv2.createMergeMertens().process(images)

    dark = images[0]
    mid = images[len(images) // 2]
    del images  # free the other brackets — only dark + mid are needed now

    masks = window_mask.get_label_masks(mid)
    seg_win = masks["window"] if masks else None
    wmask = build_window_mask(mid, seg_win)

    img = fused.astype(np.float32)
    if enhance:
        # interior finishing — windows excluded from WB and levels
        # measurement. Brightness FIRST, color cleanup AFTER: the
        # lamp/wall neutralizers key on bright pixels, so they only see
        # the casts once the image is at its final brightness.
        img = white_patch_wb(img, exclude=wmask)
        img = auto_levels(img, exclude=wmask)
        img = auto_exposure(img, target=0.70)  # AutoHDR median ~189/255
        #     (target sits below 189/255=0.74 because the S-curve and
        #      window blend later push the median up)
        img = whiten_lamps(img)
        img = neutralize_whites(
            img,
            boost=masks["wallceil"] if masks else None,
            protect=masks["floor"] if masks else None,
        )
        img = match_neutral_whites(img)
        img = s_curve(img, strength=0.05)  # measured: gentle contrast

    # window pull (optional) runs LAST, so interior brightening can't
    # wash the glass back out. wmask is computed unconditionally above
    # because WB/levels must exclude window pixels either way.
    if pull_windows:
        content = window_content(dark, wmask)
        if content is not None:
            m = wmask[..., None]
            img = img * (1 - m) + content * m

    img = sharpen_two_stage(img)
    return (img * 255).clip(0, 255).astype("uint8")
