"""Semantic region masks via SegFormer-B0 (ADE20K) and coarse scene tagging.

The model lives in ``models/segformer-b0-ade`` next to this script, so nothing is
downloaded at runtime. Masks are produced at 512px and then upsampled, because the
final LUT blend wants a soft boundary anyway - a pixel-perfect edge would show as a
hard contour once two different mappings meet.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "models", "segformer-b0-ade")

# ADE20K class ids grouped into the regions we can actually grade differently.
REGION_CLASSES = {
    "sky": [2],
    "people": [12],
    "veg": [4, 17, 9, 66, 72, 29],
    "water": [21, 26, 60, 113, 128, 109],
    "ground": [13, 6, 3, 46, 52, 91, 54, 11, 68, 94, 34, 16],
    "arch": [0, 1, 25, 48, 61, 84],
}
DEFAULT_REGIONS = ("sky", "people", "veg")

# a region is only trusted between these probabilities; below LO nothing is applied,
# above HI the region mapping is used at full weight
CONF_LO = 0.40
CONF_HI = 0.70
FEATHER_SIGMA = 1.6

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_config() -> dict:
    with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _blur2d(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(sigma * 3)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * p[i:i + a.shape[0]]
    p = np.pad(out, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * p[:, i:i + a.shape[1]]
    return out


def _resize_f(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float32 2-D array (PIL 'F' mode)."""
    im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32), mode="F")
    return np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float32)


def _resize_near(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(arr).astype(np.uint8))
    return np.asarray(im.resize(size, Image.NEAREST), dtype=np.uint8)


def _grow_from_top(mask: np.ndarray, rounds: int = 200) -> np.ndarray:
    """Connectivity flood from the top row, staying inside `mask`.

    Sky has a strong positional prior: a backlit sun scores 0.49 (below the gate) yet is
    obviously part of the sky, because it is connected to sky above it. Growing from the
    top row with a low threshold recovers those areas without lowering the gate for the
    rest of the frame. Runs at 128px, so the iterative propagation stays cheap.
    """
    reach = np.zeros_like(mask)
    reach[0, :] = mask[0, :]
    if not reach.any():
        return reach
    for _ in range(int(rounds)):
        up = np.zeros_like(reach); up[1:] = reach[:-1]
        dn = np.zeros_like(reach); dn[:-1] = reach[1:]
        lf = np.zeros_like(reach); lf[:, 1:] = reach[:, :-1]
        rt = np.zeros_like(reach); rt[:, :-1] = reach[:, 1:]
        new = (up | dn | lf | rt) & mask & ~reach
        if not new.any():
            break
        reach |= new
    return reach


def _fill_holes_bool(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    pad = np.pad((~mask).astype(np.uint8) * 255, 1, constant_values=255)
    im = Image.fromarray(np.repeat(pad[..., None], 3, axis=2))
    ImageDraw.floodfill(im, (0, 0), (128, 128, 128))
    holes = np.asarray(im)[1:-1, 1:-1, 0] == 255
    return mask | holes


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labelling with union-find; small enough to do in pure numpy loops."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    parent = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n = 0
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if not row[x]:
                continue
            up = lab[y - 1, x] if y else 0
            lf = lab[y, x - 1] if x else 0
            if up and lf:
                a, b = find(int(up)), find(int(lf))
                if a != b:
                    parent[b] = a
                lab[y, x] = a
            elif up:
                lab[y, x] = up
            elif lf:
                lab[y, x] = lf
            else:
                n += 1
                parent.append(n)
                lab[y, x] = n
    for y in range(h):
        for x in range(w):
            if lab[y, x]:
                lab[y, x] = find(int(lab[y, x]))
    return lab, n


def _fill_enclosed(mask: np.ndarray, min_area: float = 60.0,
                   min_side: int = 4, min_fill: float = 0.45) -> np.ndarray:
    """Fill enclosed holes that are *large and compact*.

    A blown-out sun is a compact blob; the gaps between bare winter branches are thin
    slivers that happen to be enclosed too. Filling every hole therefore paints the
    branches as sky — measured on a backlit sunset, sky coverage jumped from 0.30 to
    0.46 and the tree tops turned red in the mask overlay. Filtering by area and by
    bbox fill ratio keeps the sun and drops the slivers.
    """
    if not mask.any():
        return mask
    holes = _fill_holes_bool(mask) & ~mask
    if not holes.any():
        return mask
    lab, _ = _label_components(holes)
    keep = np.zeros_like(mask)
    for lid in range(1, int(lab.max()) + 1):
        ys, xs = np.nonzero(lab == lid)
        if ys.size < min_area:
            continue
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        if min(bw, bh) < min_side:
            continue
        if ys.size / float(bw * bh) < min_fill:
            continue
        keep[ys, xs] = True
    return mask | keep


def _fill_holes(g: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a soft mask.

    A blown-out sun or a backlit cloud is still sky, but the model will not label it so
    (it has no colour left to judge). Leaving it ungraded shows up as a bright patch
    sitting inside a darkened sky — measured on a backlit sunset, that was two clearly
    visible oval holes. Anything unreachable from outside the region is treated as part
    of it; ground that touches the frame border is not, so it stays untouched.
    """
    m = g > 0.5
    if not m.any():
        return g
    holes = _fill_holes_bool(m) & ~m
    if not holes.any():
        return g
    return np.maximum(g, holes.astype(np.float32))


class Segmenter:
    """Lazily-loaded SegFormer session; safe to reuse across calls."""

    _session = None
    _input_name = None
    _id2label = None
    _used_model = None

    def __init__(self, quantized: bool = True, threads: int = 0):
        if Segmenter._session is not None and Segmenter._used_model == quantized:
            return
        import onnxruntime as ort

        name = "model_quantized.onnx" if quantized else "model.onnx"
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            raise RuntimeError(f"segmentation model missing: {path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 4) - 1)
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        Segmenter._session = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        Segmenter._input_name = Segmenter._session.get_inputs()[0].name
        cfg = _load_config()
        Segmenter._id2label = {int(k): str(v) for k, v in cfg["id2label"].items()}
        Segmenter._used_model = quantized

    def _run(self, rgb: np.ndarray, size: int = 512) -> np.ndarray:
        im = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8))
        im = im.resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - _MEAN) / _STD
        x = np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)
        logits = self._session.run(None, {self._input_name: x})[0]
        logits = logits[0] if logits.ndim == 4 else logits  # (C, h, w)
        e = logits - logits.max(axis=0, keepdims=True)
        np.exp(e, out=e)
        e /= e.sum(axis=0, keepdims=True)
        return e  # (150, h, w) float32

    def masks(self, rgb: np.ndarray, regions=DEFAULT_REGIONS,
              feather: float = FEATHER_SIGMA) -> "RegionMasks":
        probs = self._run(rgb)
        h, w = rgb.shape[:2]

        masks = {}
        for name in regions:
            ids = REGION_CLASSES.get(name)
            if not ids:
                continue
            p = probs[list(ids)].sum(axis=0)  # (128, 128)
            if name == "sky":
                # sky has a positional prior: grow down from the top row with a low
                # threshold, then fill only the big compact holes (a backlit sun)
                seed = p > 0.22
                body = _grow_from_top(seed)
                if body.any():
                    body = _fill_enclosed(body)
                g = np.clip((p - CONF_LO) / (CONF_HI - CONF_LO), 0.0, 1.0)
                if body.any():
                    g = np.maximum(g, body.astype(np.float32))
            else:
                g = np.clip((p - CONF_LO) / (CONF_HI - CONF_LO), 0.0, 1.0)
                g = _fill_holes(g)
            g = _blur2d(g, feather)
            masks[name] = np.clip(g, 0.0, 1.0).astype(np.float32)

        # dominant label at 512 for scene tagging
        small = probs.reshape(probs.shape[0], -1)
        labels128 = small.argmax(axis=0).reshape(probs.shape[1:]).astype(np.uint8)
        lab_im = Image.fromarray(labels128)
        labels = np.asarray(lab_im.resize((512, 512), Image.NEAREST), dtype=np.uint8)

        return RegionMasks(masks, labels, (h, w))


class RegionMasks:
    """Per-region soft weights plus the dominant label map."""

    def __init__(self, masks: dict, labels: np.ndarray, image_hw: tuple[int, int]):
        self._masks = masks
        self.labels = labels
        self.image_hw = image_hw
        self._cache: dict = {}

    @property
    def regions(self) -> list:
        return sorted(self._masks)

    def weight(self, region: str, hw: tuple[int, int] | None = None) -> np.ndarray | None:
        """Soft 0..1 weight for a region at the given size (bilinear upsampled)."""
        if region not in self._masks:
            return None
        h, w = hw or self.image_hw
        key = (region, h, w)
        if key not in self._cache:
            self._cache[key] = _resize_f(self._masks[region], (w, h))
        return self._cache[key]

    def coverage(self) -> dict:
        """Fraction of the frame each region roughly covers (from the 512 mask)."""
        return {k: round(float((v > 0.5).mean()), 4) for k, v in sorted(self._masks.items())}

    def _skin_ratio(self, rgb: np.ndarray) -> float:
        """Fraction of the 'people' region that actually looks like skin.

        ADE20K happily labels a close-up cat as 'person' (measured: 88% of frame, with
        only 3% skin-like pixels inside it). Real portraits sit at 15-40% skin. This
        ratio is what separates the two, and it needs no extra model.
        """
        pm = self._masks.get("people")
        if pm is None:
            return 0.0
        sel = pm > 0.5
        n = int(sel.sum())
        if n < 200:
            return 0.0
        from lut_core import srgb_to_lab
        h, w = pm.shape
        im = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8))
        im = im.resize((w, h), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        lab = srgb_to_lab(a.reshape(-1, 3))
        L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
        skin = (L > 30) & (L < 88) & (A > 8) & (A < 32) & (B > 10) & (B < 38)
        return float(skin.reshape(-1)[sel.reshape(-1)].mean())

    def person_pct(self, rgb: np.ndarray) -> float:
        """Percent of frame occupied by actual people, not by the person label.

        ADE20K labels a close-up cat as "person" over 88% of the frame. A portrait
        sits at 15-40% skin inside the person mask, that cat at 3%. Below 8% skin
        the mask is not a person, so this returns 0 and callers can skip anything
        that only makes sense with a human in frame (skin hue, for one).
        """
        cov = self.coverage().get("people", 0.0)
        if cov < 0.005:
            return 0.0
        return cov * 100.0 if self._skin_ratio(rgb) >= 0.08 else 0.0

    def scene_scores(self, rgb: np.ndarray | None = None) -> dict:
        """Coarse scene type scores derived from region coverage and exposure."""
        cov = self.coverage()
        sky = cov.get("sky", 0.0)
        people = cov.get("people", 0.0)
        veg = cov.get("veg", 0.0)
        water = cov.get("water", 0.0)

        lab = self.labels
        total = float(lab.size)
        def frac(*ids):
            return float(np.isin(lab, ids).sum()) / total

        arch = frac(0, 1, 25, 48, 61, 84)
        # indoor furniture cues: bed, ceiling, table, box, sofa, shelf, curtain, wardrobe,
        # cabinet - ADE20K covers interiors with furniture, not with "indoor"
        indoor_hint = frac(5, 3, 7, 15, 18, 23, 24, 35, 41, 10, 33, 30)

        lum = 0.5
        if rgb is not None:
            small = rgb[::max(1, rgb.shape[0] // 256), ::max(1, rgb.shape[1] // 256)]
            lum = float((small[..., 0] * 0.2126 + small[..., 1] * 0.7152
                         + small[..., 2] * 0.0722).mean())

        skin_gate = 1.0
        if people > 0.02 and rgb is not None:
            # below 5% skin the "person" is almost certainly a mislabel; 20%+ is a face
            skin_gate = min(1.0, max(0.0, (self._skin_ratio(rgb) - 0.05) / 0.15))

        s = {}
        s["portrait"] = min(1.0, people * 6.0) * skin_gate
        s["landscape"] = min(1.0, (sky * 1.6 + veg * 1.2 + water * 1.5) * (1.0 if people < 0.05 else 0.4))
        s["street"] = min(1.0, arch * 1.8 + (0.4 if 0.05 < sky < 0.4 and arch > 0.15 else 0.0))
        s["indoor"] = min(1.0, indoor_hint * 2.5 + (0.5 if sky < 0.02 else 0.0))
        s["night"] = min(1.0, max(0.0, (0.30 - lum) / 0.30) * 1.4)
        s = {k: round(float(min(1.0, v)), 3) for k, v in s.items()}
        return dict(sorted(s.items(), key=lambda kv: -kv[1]))

    def tags(self, rgb=None, top: int = 2, floor: float = 0.25) -> list:
        scores = self.scene_scores(rgb)
        out = [(k, v) for k, v in scores.items() if v >= floor][:top]
        return [k for k, _ in out] or [next(iter(scores))]


_DEFAULT_SEGMENTER = None


def get_segmenter(quantized: bool = True) -> Segmenter:
    global _DEFAULT_SEGMENTER
    if _DEFAULT_SEGMENTER is None:
        _DEFAULT_SEGMENTER = Segmenter(quantized=quantized)
    return _DEFAULT_SEGMENTER
