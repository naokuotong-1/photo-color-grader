"""Reference-free picture checks.

The LUT pipeline answers "how far is this render from the target?". It cannot
answer two questions that matter just as much:

    measure(rgb)   classic scene statistics — exposure, black point, clipping,
                   white balance, skin hue, colorfulness — so a picture can be
                   judged without a reference image at all.

    transfer(rgb)  is this frame display-referred, log-encoded, or already
                   graded? Fitting a look onto log footage bakes the log curve
                   into the LUT and the error is invisible on screen.

Everything here is advisory. A night shot is *supposed* to have a low median, a
sunset is *supposed* to measure warm, a grey warehouse is *supposed* to be drab.
So each check carries a wide band, and a warning means "look at this", never
"this is wrong".

Thresholds follow the classical targets (median luminance ~0.45, black point
< 0.02, skin hue ~20 deg HSL, colorfulness with no target at all) with the bands
widened where a legitimate scene would otherwise trip them. Calibrated against
the test set in the repo: a backlit sunset, a midday street, an indoor cat.
"""
from __future__ import annotations

import math

import numpy as np

import lut_core as lc


# ---------------------------------------------------------------- primitives

def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma on the encoded values, which is what a histogram shows."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _pixels(rgb: np.ndarray, stride: int) -> np.ndarray:
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if stride > 1:
        a = a[::stride, ::stride]
    a = a.reshape(-1, 3)
    return np.clip(a[np.isfinite(a).all(axis=1)], 0.0, 1.0)


def _hsl_hue(a: np.ndarray) -> np.ndarray:
    mx = a.max(axis=1)
    mn = a.min(axis=1)
    d = mx - mn
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros(len(a), dtype=np.float32)
    nz = d > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        hr = 60.0 * (((g - b) / np.where(nz, d, 1.0)) % 6.0)
        hg = 60.0 * (((b - r) / np.where(nz, d, 1.0)) + 2.0)
        hb = 60.0 * (((r - g) / np.where(nz, d, 1.0)) + 4.0)
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb))
    h = np.where(nz, h % 360.0, 0.0)
    return h


def _skin_mask(a: np.ndarray) -> np.ndarray:
    """YCbCr skin locus, not an HSV box.

    An HSV box that accepts skin also accepts wood, khaki and amber practicals.
    The locus tolerance is deliberately wide toward flushed/sunburnt skin and
    narrow toward the wood/amber direction, because that is the error this check
    exists to catch.
    """
    y = 0.299 * a[:, 0] + 0.587 * a[:, 1] + 0.114 * a[:, 2]
    cb = (a[:, 2] - y) * 0.564 + 0.5
    cr = (a[:, 0] - y) * 0.713 + 0.5
    return (cb > 0.30) & (cb < 0.51) & (cr > 0.52) & (cr < 0.67) & (y > 0.12) & (y < 0.92)


def _hasler_colorfulness(a: np.ndarray) -> float:
    """Hasler-Suesstrunk colourfulness, reported on the 0-255 scale."""
    r, g, b = a[:, 0] * 255.0, a[:, 1] * 255.0, a[:, 2] * 255.0
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


# ---------------------------------------------------------------- measurement

def measure(rgb: np.ndarray, stride: int = 4, person_pct: float | None = None) -> dict:
    """Scene statistics for one picture. No reference image needed.

    Returns nested groups plus a ``warnings`` list; every entry is advisory and
    names the metric that tripped. Safe to call on the source *and* on a render:
    comparing the two is how a bad LUT gets caught before delivery.

    person_pct: percentage of frame covered by people, from the segmentation
    model. Skin hue is only checked when a person is actually present — colour
    alone cannot tell skin from a warm wall. Measured on a sunset street, the
    YCbCr locus alone calls 30% of the frame "skin" (buildings, sunlight, awnings),
    so without this gate the warning fires on scenes that contain nobody.
    """
    a = _pixels(rgb, stride)
    if len(a) < 256:
        return {"error": "too few pixels to measure"}

    Y = _luma(a)
    # p0.5 / p99.9 rather than p1 / p99: a backlit frame has a clipped sun in
    # well under 1% of pixels, and clipping it out of the range makes a perfectly
    # display-referred sunset look like it is "holding highlights back".
    p1, p50, p99 = (float(v) for v in np.percentile(Y, [0.5, 50, 99.9]))
    mx = a.max(axis=1)
    mn = a.min(axis=1)
    clipped = float((mn >= 0.985).mean())
    crushed = float((mx <= 0.015).mean())
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    mean_sat = float(sat.mean())

    # Shades-of-Gray (Minkowski p=6) on unclipped pixels. Excluding clipped ones
    # matters: a blown sky is white regardless of the illuminant.
    usable = mn < 0.985
    if usable.sum() >= 256:
        mink = (np.mean(a[usable] ** 6, axis=0)) ** (1.0 / 6.0)
        gains = mink / max(float(mink.mean()), 1e-6)
    else:
        gains = np.ones(3, dtype=np.float32)
    cast_stops = float(math.log2(max(float(gains.max()), 1e-6) / max(float(gains.min()), 1e-6)))

    # A cast is only trustworthy where the scene actually contains near-neutrals.
    # A sunset has almost none, so this gate keeps it from crying wolf.
    lab = lc.srgb_to_lab(a[:: max(1, len(a) // 20000)])
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    neu = chroma < 10.0
    neutral_pct = float(neu.mean()) * 100.0
    neutral_ab = (float(np.median(lab[neu, 1])), float(np.median(lab[neu, 2]))) if neu.sum() >= 32 else (0.0, 0.0)

    skin = _skin_mask(a)
    skin_pct = float(skin.mean()) * 100.0
    skin_hue = float(np.median(_hsl_hue(a[skin]))) if skin.sum() >= 256 else None

    out = {
        "exposure": {
            "median_luma": round(p50, 4),
            "p1": round(p1, 4),
            "p99": round(p99, 4),
            "dynamic_range": round(p99 - p1, 4),
        },
        "clipping": {"clipped_pct": round(clipped * 100, 2),
                     "crushed_pct": round(crushed * 100, 2)},
        "white_balance": {
            "cast_stops": round(cast_stops, 3),
            "gains": [round(float(g), 3) for g in gains],
            "neutral_pixels_pct": round(neutral_pct, 1),
        },
        "skin": {"pixels_pct": round(skin_pct, 1),
                 "hue_deg": None if skin_hue is None else round(skin_hue, 1),
                 "gated_by_person": person_pct is not None},
        "colorfulness": round(_hasler_colorfulness(a), 1),
        "mean_saturation": round(mean_sat, 3),
    }
    out["warnings"] = _warnings(out, neutral_ab, person_pct)
    return out


def _warnings(m: dict, neutral_ab: tuple[float, float], person_pct: float | None) -> list:
    w = []

    def add(metric, value, note):
        w.append({"metric": metric, "value": value, "note": note})

    clip = m["clipping"]["clipped_pct"]
    if clip > 3.0:
        add("clipped_pct", clip, "高光成片死白，过曝区没有可恢复的颜色信息")
    crush = m["clipping"]["crushed_pct"]
    # 10%, not 3%: a backlit silhouette legitimately has 4-5% near-black trees,
    # and a night shot more. Rendering *into* new crush is caught by compare().
    if crush > 10.0:
        add("crushed_pct", crush, "大片纯黑，暗部细节被压没了")

    wb = m["white_balance"]
    # Only complain when near-neutrals exist AND they lean strongly one way.
    if wb["neutral_pixels_pct"] >= 2.0:
        a_, b_ = neutral_ab
        if wb["cast_stops"] > 0.6:
            add("cast_stops", wb["cast_stops"],
                f"整体色偏约 {wb['cast_stops']:.2f} 档；若画面本应有中性色，说明白平衡有偏")
        if abs(b_) > 14:
            add("neutral_b", round(b_, 1),
                "本该中性的像素明显偏蓝/偏黄（b* 偏大），不是风格而是色偏")

    skin = m["skin"]
    # No person mask means the colour test is unreliable, so it stays silent
    # rather than guessing. Teeth: measured 30% of a street scene reads "skin".
    if person_pct is None:
        skin["note"] = "no person mask available, skin hue not judged"
    elif person_pct < 0.5:
        skin["note"] = "no person in frame, skin hue not judged"
    elif skin["hue_deg"] is not None and skin["pixels_pct"] >= 1.5:
        d = skin["hue_deg"] - 20.0
        if abs(d) > 12:
            add("skin_hue", skin["hue_deg"],
                f"肤色色相 {skin['hue_deg']:.0f}° 偏离 20° 目标 {d:+.0f}°，皮肤会发红或发绿")

    c = m["colorfulness"]
    if c < 4.0:
        add("colorfulness", c, "几乎无色，确认这是黑白风格而不是把饱和度做没了")
    if c > 110.0:
        add("colorfulness", c, "色彩浓到超出常见范围，检查是否过饱和")
    return w


def compare(before: dict, after: dict, label: str = "render") -> list:
    """Diff two ``measure`` results and report what the render did to the image.

    This is the part that catches a bad LUT before delivery: a style that is
    correct on the training pair can still crush blacks on a night shot.
    """
    if "exposure" not in before or "exposure" not in after:
        return []
    w = []
    d_med = after["exposure"]["median_luma"] - before["exposure"]["median_luma"]
    if abs(d_med) > 0.22:
        w.append({"metric": f"{label}.median_luma", "value": round(d_med, 3),
                  "note": f"整体亮度变化 {d_med:+.2f}，幅度接近一档曝光"})

    d_clip = after["clipping"]["clipped_pct"] - before["clipping"]["clipped_pct"]
    if d_clip > 1.5:
        w.append({"metric": f"{label}.clipped_pct", "value": round(d_clip, 2),
                  "note": f"新增 {d_clip:.1f}% 高光死白，这个风格会烧掉亮部"})

    d_crush = after["clipping"]["crushed_pct"] - before["clipping"]["crushed_pct"]
    if d_crush > 1.5:
        w.append({"metric": f"{label}.crushed_pct", "value": round(d_crush, 2),
                  "note": f"新增 {d_crush:.1f}% 纯黑，暗部被压死"})

    bs, as_ = before["mean_saturation"], after["mean_saturation"]
    if bs > 0.05:
        rel = (as_ - bs) / bs
        if rel < -0.40:
            w.append({"metric": f"{label}.saturation", "value": round(rel, 3),
                      "note": f"饱和度掉了 {-rel:.0%}"})
        elif rel > 0.60:
            w.append({"metric": f"{label}.saturation", "value": round(rel, 3),
                      "note": f"饱和度涨了 {rel:.0%}，注意是否过饱和"})
    return w


# ---------------------------------------------------------------- transfer

def transfer(rgb: np.ndarray, stride: int = 4) -> dict:
    """Decide whether a frame is display-referred, log, or already graded.

    Reads the *shape* of the encoding, not metadata, because by the time a frame
    reaches this tool it is usually a PNG or a libraw render. Three outcomes, not
    two: a heavily flattened, lifted, desaturated display-referred grade is
    genuinely indistinguishable from log by shape alone, and guessing either way
    is worse than saying so.
    """
    m = measure(rgb, stride)
    if "exposure" not in m:
        return {"verdict": "unknown", "error": m.get("error")}

    lo = m["exposure"]["p1"]
    hi = m["exposure"]["p99"]
    med = m["exposure"]["median_luma"]
    spread = m["exposure"]["dynamic_range"]
    sat = m["mean_saturation"]
    pos = (med - lo) / max(spread, 1e-6)

    # Thresholds calibrated so that ordinary display-referred frames score zero.
    # Measured: sunset street (0.035/0.994/0.96), backlit sunset (0.001/1.0/1.0),
    # indoor cat (0.005/0.945/0.94), a graded photo (0.035/0.896/0.86).
    # The midpoint-position signal was dropped: a normally exposed photo has the
    # median at 0.64 of its range, which is indistinguishable from log.
    hits = []
    if lo > 0.055:
        hits.append("black lifted")
    if hi < 0.88:
        hits.append("highlight held back")
    if spread < 0.72:
        hits.append("narrow range")
    if sat < 0.22:
        hits.append("desaturated")

    n = len(hits)
    # Two signals, not one, before saying anything. A single lifted black is also
    # what a synthetic test image or a foggy morning looks like; measured on the
    # repo's synthetic pairs (p0.5 = 0.08), one-signal "ambiguous" would have
    # flagged every fit. Three signals means the whole histogram shape agrees.
    verdict = "log" if n >= 3 else ("ambiguous" if n == 2 else "display")
    out = {
        "verdict": verdict,
        "signals": hits,
        "range": [round(lo, 3), round(hi, 3)],
        "median_position": round(pos, 3),
        "mean_saturation": round(sat, 3),
    }
    if verdict == "log":
        out["advice"] = ("看起来是 log / 未转色的素材。直接 fit 会把 log 曲线本身学进 LUT，"
                         "画面看着合理但是错的。先用相机转换 LUT 转到显示参考，或导 16bit TIFF。")
    elif verdict == "ambiguous":
        out["advice"] = ("低对比、抬黑位、低饱和，既可能是 log 也可能是褪色风格的成片。"
                         "确认来源后再 fit；成片可以继续，log 需要先转色。")
    return out
