"""photo-color-grader CLI.

Commands
    fit      learn a style LUT from (original, retouched) pairs
    ref      derive a style LUT from a reference image applied to a target image
    apply    render images through a stored style (batch capable)
    verify   LUT self-check, or fit quality against a target image
    check    reference-free check: what kind of frame is this, and is it broken?
    styles   list / inspect stored styles
    export   write the .cube files (and node-wiring notes)
    undo     drop one batch and refit from the rest
    batches  list the archived batches of a style

Every command prints a JSON object on stdout so agents can chain them.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lut_core as lc
import style_store as ss
import imgcheck as chk

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".raw"}
EXT |= RAW_EXT

try:
    import tifffile
except Exception:  # optional, only for true 16-bit RGB
    tifffile = None

try:
    import seg as segmod
except Exception:  # optional: no onnxruntime / no model -> global grading only
    segmod = None


# ---------------------------------------------------------------- io


def load_raw(path: str) -> np.ndarray:
    """Decode a RAW to sRGB-encoded float — deliberately approximate.

    This is *not* a replacement for a proper camera profile. libraw applies the
    camera's white balance and a generic colour matrix, so the same style applied to
    an ARW and a CR2 will not match what Lightroom would give. Good enough for "just
    grade the raw", not good enough to be the reference rendering.
    """
    try:
        import rawpy
    except ImportError:
        raise SystemExit(
            f"reading {os.path.splitext(path)[1]} needs rawpy: python -m pip install rawpy\n"
            "or export a 16-bit TIFF from Lightroom / Capture One (recommended)"
        )
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(2.4, 12.92),
            user_flip=None,
        )
    return (rgb.astype(np.float32) / 65535.0)


def load_rgb(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        return load_raw(path)
    if ext in (".tif", ".tiff") and tifffile is not None:
        a = np.asarray(tifffile.imread(path))
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        if a.shape[2] > 3:
            a = a[..., :3]
        if a.dtype == np.uint16:
            return a.astype(np.float32) / 65535.0
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
        return np.clip(a.astype(np.float32), 0.0, 1.0)

    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im).astype(np.float32)
        scale = 65535.0 if a.max() > 255 else 255.0
        return np.repeat((a / scale)[..., None], 3, axis=2)
    return np.asarray(im.convert("RGB")).astype(np.float32) / 255.0


LOSSLESS_EXT = (".png", ".tif", ".tiff", ".bmp", ".webp")


def read_meta(path: str) -> dict:
    """ICC / EXIF / DPI of the source, so the render keeps the same colour contract.

    Losing the ICC profile is silent and expensive: a P3 or Adobe RGB frame written
    as untagged RGB gets read back as sRGB by every viewer, which looks flat and
    undersaturated -- indistinguishable from "it got compressed".
    """
    meta = {"icc": None, "exif": None, "dpi": None}
    if os.path.splitext(path)[1].lower() in RAW_EXT:
        return meta
    try:
        with Image.open(path) as im:
            meta["icc"] = im.info.get("icc_profile")
            meta["dpi"] = im.info.get("dpi")
            raw_exif = im.info.get("exif")
            if raw_exif:
                try:
                    ex = Image.Exif()
                    ex.load(raw_exif)
                    ex.pop(0x0112, None)  # pixels are already transposed on load
                    meta["exif"] = ex.tobytes()
                except Exception:
                    meta["exif"] = None
    except Exception:
        pass
    return meta


def save_rgb(path: str, arr: np.ndarray, bits: int = 8, quality: int = 98,
             meta: dict | None = None, dither: bool | None = None) -> str:
    arr = np.clip(arr, 0.0, 1.0)
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = meta or {}
    icc, exif, dpi = meta.get("icc"), meta.get("exif"), meta.get("dpi")

    if bits == 16 and ext in (".tif", ".tiff"):
        if tifffile is None:
            raise SystemExit("16-bit output needs the `tifffile` package (pip install tifffile)")
        tkw = {"icc_profile": icc} if icc else {}
        tifffile.imwrite(path, np.rint(arr * 65535.0).astype(np.uint16), **tkw)
        return "16bit-tiff"

    lossy = ext in (".jpg", ".jpeg")
    if dither is None:
        # dither exists to stop banding when a smooth gradient is squeezed into 8
        # bits. JPEG quantisation is already an order of magnitude coarser than
        # 1 LSB, so dithering there only adds noise and costs bitrate.
        dither = not lossy
    if dither:
        q = arr * 255.0 + (np.random.default_rng(0).random(arr.shape) - 0.5)
        note = "8bit+dither"
    else:
        q = arr * 255.0
        note = "8bit"
    im = Image.fromarray(np.clip(np.rint(q), 0, 255).astype(np.uint8))

    kw: dict = {}
    if lossy:
        kw = {"quality": int(quality), "subsampling": 0, "optimize": True}
        note += f"+jpeg q{int(quality)} 4:4:4"
    elif ext == ".png":
        kw = {"optimize": True}
    elif ext == ".webp":
        kw = {"lossless": True, "quality": 100}
    if icc:
        kw["icc_profile"] = icc
    if exif:
        kw["exif"] = exif
    if dpi:
        kw["dpi"] = dpi
    im.save(path, **kw)
    return note


def iter_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in EXT:
                out.append(os.path.join(root, f))
    return sorted(out)


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def resize_long(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Downscale so the long edge fits; float path (no 8-bit detour)."""
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    sc = long_edge / max(h, w)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    chans = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32), mode="F")
            .resize((nw, nh), Image.BILINEAR),
            dtype=np.float32,
        )
        for c in range(3)
    ]
    return np.stack(chans, axis=-1)


def parse_regions(spec: str) -> list:
    if not spec:
        return []
    known = set(segmod.REGION_CLASSES) if segmod else set()
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if part not in known:
            raise SystemExit(f"unknown region '{part}' (choose from {sorted(known)})")
        out.append(part)
    return out


def collect_region_samples(loaded, regions, seg, long_edge=1024, max_per_region=120000):
    """Gather (source, retouched) pixel pairs inside each region, on a downscaled copy.

    Region residuals only need enough colour coverage, not full resolution; 1024px
    keeps the colour range while making segmentation 20x cheaper.
    """
    pool = {r: ([], []) for r in regions}
    for src, tgt in loaded:
        s_small = resize_long(src, long_edge)
        t_small = resize_long(tgt, long_edge)
        hw = s_small.shape[:2]
        m = seg.masks(s_small, regions=regions)
        flat_s = s_small.reshape(-1, 3)
        flat_t = t_small.reshape(-1, 3)
        for r in regions:
            w = m.weight(r, hw)
            if w is None:
                continue
            sel = w.reshape(-1) > 0.55
            n = int(sel.sum())
            if n < 2000:
                continue
            idx = np.flatnonzero(sel)
            if n > max_per_region:
                idx = idx[np.linspace(0, n - 1, max_per_region).astype(np.int64)]
            pool[r][0].append(flat_s[idx])
            pool[r][1].append(flat_t[idx])
    out = {}
    for r, (a, b) in pool.items():
        out[r] = (np.concatenate(a), np.concatenate(b)) if a else (None, None)
    return out


def nearest_style(ws: str, lut: np.ndarray, threshold: float):
    """Return (name_or_None, best_distance, top_three) for the closest stored style."""
    cands = []
    for s in ss.list_styles(ws):
        try:
            other, _ = ss.load_lut(ws, s["style_id"])
        except SystemExit:
            continue
        if other.shape != lut.shape:
            continue
        cands.append((s["style_id"], lc.style_distance(lut, other)))
    if not cands:
        return None, None, []
    cands.sort(key=lambda kv: kv[1])
    best_name, best_d = cands[0]
    top = cands[:3]
    return (best_name if best_d < threshold else None), best_d, top


def region_extra_summary(base_measured: dict, region_measured: dict) -> str:
    """What a region LUT does *on top of* the global layer, in one line."""
    d = {k: region_measured[k] - base_measured[k] for k in base_measured}
    bits = []
    if abs(d["exposure_ev"]) >= 0.02:
        bits.append(f"{'再提亮' if d['exposure_ev'] > 0 else '再压暗'} {abs(d['exposure_ev']):.2f} 档")
    if abs(d["contrast_slope"]) >= 0.01:
        bits.append(f"对比{'再增' if d['contrast_slope'] > 0 else '再减'} {abs(d['contrast_slope']) * 100:.0f}%")
    if abs(d["highlight_L_delta"]) >= 1.0:
        bits.append(f"高光{'再抬' if d['highlight_L_delta'] > 0 else '再收'} {abs(d['highlight_L_delta']):.1f} L*")
    if abs(d["shadow_L_delta"]) >= 1.0:
        bits.append(f"暗部{'再抬' if d['shadow_L_delta'] > 0 else '再压'} {abs(d['shadow_L_delta']):.1f} L*")
    if abs(d["saturation_ratio"]) >= 0.02:
        bits.append(f"饱和{'再升' if d['saturation_ratio'] > 0 else '再降'} {abs(d['saturation_ratio']) * 100:.0f}%")
    if abs(d["grey_b_shift"]) >= 0.8:
        bits.append(f"再{'偏黄' if d['grey_b_shift'] > 0 else '偏蓝'} {abs(d['grey_b_shift']):.1f}")
    return "；".join(bits) if bits else "与全局层几乎一致"


def auto_name(ws: str) -> str:
    base = "look_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name, i = base, 1
    while os.path.exists(os.path.join(ws, name)):
        i += 1
        name = f"{base}_{i}"
    return name


def tag_scenes(loaded, seg, limit: int = 6) -> dict:
    """Average scene-type scores over a few source images.

    Only a handful is enough: a batch of one session shares its subject, and running
    the segmenter over every frame would double the cost of `fit`.
    """
    acc: dict = {}
    n = 0
    for src, _tgt in loaded[:limit]:
        small = resize_long(src, 1024)
        m = seg.masks(small, regions=("sky", "people", "veg", "water", "ground", "arch"))
        for k, v in m.scene_scores(small).items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    if n == 0:
        return {}
    return {k: round(v / n, 3) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}


# ---------------------------------------------------------------- metrics


KEY_QUANTILES = (1, 10, 50, 90, 99)


def image_key(img: np.ndarray) -> dict:
    """Tonal key of a frame: luma quantiles in the same encoded space the LUT lives in.

    A look is not exposure-independent. The one fitted here lifts midtones about 1.4
    stops; handing it a frame that is already 1.5 stops brighter than the reference
    drove the median from 0.55 to 0.95 and blew the picture out, so the LUT alone
    cannot carry the style across differently exposed material.
    """
    y = chk._luma(img)
    return {f"p{q}": round(float(np.percentile(y, q)), 4) for q in KEY_QUANTILES}


def _mean_key(keys: list) -> dict:
    if not keys:
        return {}
    return {k: round(float(np.mean([x[k] for x in keys])), 4) for k in keys[0]}


def key_stops(input_key: dict, ref_key: dict) -> float:
    m_in = max(float(input_key.get("p50", 0.0)), 1e-4)
    m_ref = max(float(ref_key.get("p50", 0.0)), 1e-4)
    return float(np.log2(m_ref / m_in))


def match_key(img: np.ndarray, gamma: float, lo: float = 0.03, hi: float = 25.0) -> np.ndarray:
    """Put the frame's luma key where the style was learned, before the LUT.

    This is what a colourist does — set the level, then the look. The gain is computed
    on luma and handed to RGB as one multiplier, so channel ratios (and therefore
    saturation) survive; a per-channel gamma would darken by desaturating, which is
    the wrong direction for flat material. The gain curve is smoothed and clamped so
    the shadow end cannot blow up noise.
    """
    y = chk._luma(img).astype(np.float32)
    grid = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    k = np.power(np.maximum(grid, 1e-3), np.float32(gamma - 1.0))
    k = np.clip(k, lo, hi)
    k = np.convolve(np.pad(k, 3, mode="edge"), np.ones(7, dtype=np.float32) / 7, mode="valid")
    gain = np.interp(y.ravel(), grid, k).reshape(y.shape).astype(np.float32)
    return np.clip(img * gain[..., None], 0.0, 1.0)


def color_metrics(out: np.ndarray, ref: np.ndarray, stride: int = 7) -> dict:
    if out.shape != ref.shape:
        return {"error": f"shape mismatch {out.shape} vs {ref.shape}"}
    a = out[::stride, ::stride].reshape(-1, 3)
    b = ref[::stride, ::stride].reshape(-1, 3)
    de = lc.delta_e2000(lc.srgb_to_lab(a), lc.srgb_to_lab(b))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    psnr = float("inf") if rmse == 0 else 20 * np.log10(1.0 / rmse)
    same_geometry = out.shape == ref.shape
    return {
        "dE2000_mean": round(float(de.mean()), 3),
        "dE2000_p95": round(float(np.percentile(de, 95)), 3),
        "dE2000_max": round(float(de.max()), 3),
        "psnr_db": "inf" if not np.isfinite(psnr) else round(psnr, 2),
        "geometry_preserved": bool(same_geometry),
    }


def lut_health(lut: np.ndarray) -> dict:
    n = lut.shape[0]
    ident = lc.identity_lut(n)
    viol = 0
    for ax in range(3):
        viol += int((np.diff(lut[..., ax], axis=ax) < -1e-6).sum())
    return {
        "size": n,
        "max_identity_dev": round(float(np.abs(lut - ident).max()), 4),
        "mean_identity_dev": round(float(np.abs(lut - ident).mean()), 4),
        "monotonic_violations": viol,
        "corner_black": [round(float(v), 3) for v in lut[0, 0, 0]],
        "corner_white": [round(float(v), 3) for v in lut[-1, -1, -1]],
    }


# ---------------------------------------------------------------- commands


def cmd_fit(a):
    ws = ss.resolve_workspace(a)
    pairs = []
    if a.pair:
        for s, t in a.pair:
            pairs.append((s, t))
    elif a.src and a.tgt:
        src_files = {stem(p): p for p in iter_images(a.src)}
        tgt_files = {stem(p): p for p in iter_images(a.tgt)}
        common = sorted(set(src_files) & set(tgt_files))
        if not common:
            raise SystemExit("no file pairs matched by stem between --src and --tgt")
        pairs = [(src_files[k], tgt_files[k]) for k in common]
    else:
        raise SystemExit("provide --pair SRC TGT (repeatable) or both --src and --tgt")

    loaded = [(load_rgb(s), load_rgb(t)) for s, t in pairs]
    regions = parse_regions(a.regions)

    # Guard the input before it becomes a LUT. Fitting a look onto log footage
    # teaches the LUT the log curve itself: the render looks plausible and is
    # wrong. Advisory only — the user may know the material better than we do.
    input_guard = []
    for (s_path, _), (img, _t) in zip(pairs, loaded):
        tr = chk.transfer(img)
        if tr["verdict"] != "display":
            input_guard.append({"file": os.path.basename(s_path), **tr})

    # fresh samples from this batch, kept for the archive and for refitting
    rng = np.random.default_rng(11)
    new_s = [lc._subsample_pair(s, t, a.archive_samples, rng) for s, t in loaded]
    batch_S = np.concatenate([x[0] for x in new_s], axis=0)
    batch_T = np.concatenate([x[1] for x in new_s], axis=0)

    lut, diag = lc.fit_lut(
        [(batch_S, batch_T)],
        size=a.size,
        samples_per_pair=a.samples,
        monotonic=a.monotonic,
        smooth_passes=a.smooth,
    )

    region_luts: dict = {}
    region_diag: dict = {}
    seg_info: dict = {}
    batch_regions: dict = {}
    seg = None

    if regions and segmod is not None:
        try:
            seg = segmod.get_segmenter(quantized=a.seg_quantized)
            samples = collect_region_samples(loaded, regions, seg)
            for r in regions:
                s_r, t_r = samples.get(r, (None, None))
                if s_r is None or len(s_r) < a.min_region_pixels:
                    region_diag[r] = {"skipped": "too few pixels",
                                      "pixels": 0 if s_r is None else int(len(s_r))}
                    continue
                # keep the whole regional sample set: it is already capped by
                # collect_region_samples, and truncating it would make an undo land on
                # a slightly different regional LUT than the version it is undoing
                batch_regions[r] = (s_r, t_r)
                rlut, rdiag = lc.fit_lut(
                    [(s_r, t_r)],
                    size=a.size,
                    samples_per_pair=int(min(a.samples, len(s_r))),
                    monotonic=a.monotonic,
                    smooth_passes=a.smooth,
                    baseline=lut,
                    residual_scale=a.region_strength,
                    seed=1,
                )
                dev = float(np.abs(rlut - lut).max())
                if dev < 0.004:
                    region_diag[r] = {"skipped": "no residual learned",
                                      "pixels": int(len(s_r)), "max_dev_from_base": round(dev, 4)}
                    continue
                region_luts[r] = rlut
                region_diag[r] = {
                    "pixels": int(len(s_r)),
                    "max_dev_from_base": round(dev, 4),
                    "coverage_pct": rdiag.get("coverage_pct"),
                    "clipped_pct_observed": rdiag.get("lut_clipped_pct_observed"),
                }
            seg_info = {
                "model": "segformer-b0-ade" + ("-q" if a.seg_quantized else ""),
                "requested": regions,
                "active": sorted(region_luts),
            }
        except Exception as exc:
            region_diag["_error"] = f"{type(exc).__name__}: {exc}"
    elif regions:
        region_diag["_error"] = "segmentation unavailable (onnxruntime or model missing)"

    # ------------------------------------------------ which style does this belong to
    auto_info = None
    existing = None
    if a.name:
        name = a.name
        existing = os.path.exists(os.path.join(ws, name, "style.json"))
        decision = "merge-into-named" if existing else "create-named"
    else:
        cand, dist, top = nearest_style(ws, lut, a.auto_threshold)
        name = cand or auto_name(ws)
        existing = cand is not None
        decision = "merge-nearest" if existing else "create-new"
        auto_info = {"decision": decision, "style": name,
                     "distance": None if dist is None else round(dist, 3),
                     "threshold": a.auto_threshold,
                     "nearest": [[n, round(d, 3)] for n, d in top]}

    merged_from = 0
    if existing:
        # refit from every archived batch instead of stacking LUT composites: that is
        # what keeps a style from drifting after ten incremental updates
        hist = ss.merge_batch_samples(ws, name, region_names=regions)
        if "S" in hist and len(hist["S"]):
            merged_from = len(hist["S"])
            all_s = np.concatenate([hist["S"], batch_S], axis=0)
            all_t = np.concatenate([hist["T"], batch_T], axis=0)
            lut, diag = lc.fit_lut(
                [(all_s, all_t)],
                size=a.size,
                samples_per_pair=a.samples,
                monotonic=a.monotonic,
                smooth_passes=a.smooth,
            )
            if regions and seg is not None:
                for r in list(region_luts):
                    rh = hist.get(r)
                    if rh is None:
                        continue
                    rb = batch_regions.get(r)
                    rs = np.concatenate([rh[0], rb[0]]) if rb else rh[0]
                    rt = np.concatenate([rh[1], rb[1]]) if rb else rh[1]
                    rlut, _ = lc.fit_lut(
                        [(rs, rt)], size=a.size,
                        samples_per_pair=int(min(a.samples, len(rs))),
                        monotonic=a.monotonic, smooth_passes=a.smooth,
                        baseline=lut, residual_scale=a.region_strength, seed=1,
                    )
                    region_luts[r] = rlut

    def render(img):
        if not region_luts:
            return lc.apply_lut(img, lut)
        m = segmod.get_segmenter(quantized=a.seg_quantized).masks(img, regions=list(region_luts))
        masks = {r: m.weight(r, img.shape[:2]) for r in region_luts}
        return lc.apply_lut_regional(img, lut, region_luts, masks)

    scores = [color_metrics(render(s), t) for s, t in loaded[:8]]
    agg = {
        "dE2000_mean": round(float(np.mean([m["dE2000_mean"] for m in scores])), 3),
        "dE2000_p95_mean": round(float(np.mean([m["dE2000_p95"] for m in scores])), 3),
    }
    if agg["dE2000_mean"] > a.max_de:
        sys.stderr.write(
            f"WARN: training dE2000 {agg['dE2000_mean']} exceeds --max-de {a.max_de}; "
            "the pairs may not be a consistent colour-only transform\n"
        )

    params = {"mode": "fit", "pairs": len(pairs), "samples": a.samples,
              "monotonic": a.monotonic, "lut_space": ss.LUT_SPACE,
              "regions": sorted(region_luts) if region_luts else [],
              "region_strength": a.region_strength,
              "batches_merged": len(ss.load_batches(ws, name)) if existing else 0}
    desc = lc.describe_lut(lut)
    if region_luts:
        desc["note"] = "measured 描述的是全局基础层；实际渲染会在对应区域叠加下面的分区层"
        desc["region_extra"] = {
            r: region_extra_summary(desc["measured"], lc.describe_lut(rl)["measured"])
            for r, rl in region_luts.items()
        }
    saved = ss.save_version(ws, name, lut, params,
                            {**diag, **agg, "regions": region_diag},
                            region_luts=region_luts,
                            extra={"description": desc,
                                   # where this look expects to be applied. apply uses it
                                   # to bring a differently exposed frame back to the
                                   # level the fit was learned at.
                                   "key": {"src": _mean_key([image_key(s) for s, _ in loaded]),
                                           "tgt": _mean_key([image_key(t) for _, t in loaded])}})

    batch_id = a.batch_id or datetime.datetime.now().strftime("b%Y%m%d_%H%M%S")
    samples = {"base": (batch_S, batch_T), **batch_regions}
    ss.save_batch(ws, name, batch_id, samples,
                  {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "version": saved["version"], "pairs": len(pairs),
                   "source": a.src or "(explicit pairs)",
                   "dE2000_mean": agg["dE2000_mean"],
                   "regions": sorted(batch_regions)})

    out = {"ok": True, "command": "fit", **saved, "decision": decision,
           "batch_id": batch_id, "diagnostics": diag,
           "quality": agg, "regions": seg_info, "region_diagnostics": region_diag,
           "description": desc, "health": lut_health(lut)}
    if input_guard:
        out["input_guard"] = {
            "warnings": input_guard,
            "blocking": False,
            "note": "这些原图不是标准显示参考编码。若确实是 log / 未转色素材，先转色再 fit；"
                    "若只是低对比风格成片，可以忽略。",
        }

    tag_info = {}
    if segmod is not None:
        try:
            segt = seg if seg is not None else segmod.get_segmenter(quantized=a.seg_quantized)
            scores = tag_scenes(loaded, segt)
            if scores:
                tag_info = ss.update_tags(ws, name, scores, weight=float(len(pairs)))
        except Exception as exc:
            tag_info = {"_error": f"{type(exc).__name__}: {exc}"}
    if tag_info:
        out["tags"] = tag_info

    if auto_info:
        out["auto"] = auto_info
    if merged_from:
        out["learned_from"] = {"archived_samples": int(merged_from),
                               "new_samples": int(len(batch_S))}
    out["undo_hint"] = f"grade.py undo --style {name} --batch {batch_id}"
    return out


def cmd_ref(a):
    ws = ss.resolve_workspace(a)
    ref = load_rgb(a.ref)
    tgt = load_rgb(a.target)
    lut, diag = lc.lut_from_reference(ref, tgt, size=a.size, strength=a.strength,
                                      skin_protect=a.skin, bins=a.bins)
    out = lc.apply_lut(tgt, lut)
    # how close did we land to the reference's colour distribution
    lab_o = lc.srgb_to_lab(out.reshape(-1, 3)[:: max(1, out.size // 400000)])
    lab_r = lc.srgb_to_lab(ref.reshape(-1, 3)[:: max(1, ref.size // 400000)])
    shift_before = float(np.abs(lc.srgb_to_lab(tgt.reshape(-1, 3)).mean(0)
                                - lc.srgb_to_lab(ref.reshape(-1, 3)).mean(0)).mean())
    shift_after = float(np.abs(lab_o.mean(0) - lab_r.mean(0)).mean())

    params = {"mode": "reference", "strength": a.strength, "skin_protect": a.skin,
              "bins": a.bins, "lut_space": ss.LUT_SPACE}
    saved = ss.save_version(ws, a.name, lut, params,
                            {**diag, "lab_mean_shift_before": round(shift_before, 2),
                             "lab_mean_shift_after": round(shift_after, 2)},
                            extra={"key": {"src": image_key(tgt), "tgt": image_key(out)}})
    result = {"ok": True, "command": "ref", **saved, "diagnostics": diag,
              "lab_mean_shift_before": round(shift_before, 2),
              "lab_mean_shift_after": round(shift_after, 2),
              "health": lut_health(lut)}
    if a.out:
        bits = 16 if a.bits == 16 else 8
        save_rgb(a.out, out, bits=bits, quality=getattr(a, "quality", 98),
                 meta=read_meta(a.target))
        result["rendered"] = a.out
    return result


def cmd_apply(a):
    ws = ss.resolve_workspace(a)
    auto_choice = None

    if a.style == "auto":
        if segmod is None:
            raise SystemExit("--style auto needs segmentation (onnxruntime + model)")
        probe = [load_rgb(f) for f in iter_images(a.input)[:4]]
        if not probe:
            raise SystemExit(f"no images found at {a.input}")
        segt = segmod.get_segmenter(quantized=a.seg_quantized)
        scene = tag_scenes([(p, p) for p in probe], segt, limit=4)
        ranked = []
        for s in ss.list_styles(ws):
            ts = s.get("tag_scores") or {}
            if not ts:
                continue
            ranked.append((s["style_id"],
                           float(sum(scene.get(k, 0.0) * v for k, v in ts.items()))))
        ranked.sort(key=lambda kv: -kv[1])
        if not ranked:
            raise SystemExit("no style carries scene tags yet; run fit once, or pass --style")
        a.style = ranked[0][0]
        auto_choice = {"picked": a.style, "input_scene": scene,
                       "ranking": [[n, round(v, 3)] for n, v in ranked[:3]]}

    lut, _meta = ss.load_lut(ws, a.style, a.version)
    region_luts, entry = ss.load_region_luts(ws, a.style, a.version)
    lut = ss.blended_lut(lut, a.strength)
    # region LUTs are absolute maps, so they get the same partial-strength blend;
    # otherwise the regional pixels would jump to full strength while the rest fades
    region_luts = {k: ss.blended_lut(v, a.strength) for k, v in region_luts.items()}

    segm = None
    region_note = None
    if region_luts:
        if segmod is None:
            region_note = "segmentation unavailable; fell back to the global LUT"
            region_luts = {}
        else:
            segm = segmod.get_segmenter(quantized=a.seg_quantized)

    files = iter_images(a.input)
    if not files:
        raise SystemExit(f"no images found at {a.input}")

    out_has_ext = bool(os.path.splitext(a.out)[1])
    out_is_dir = (os.path.isdir(a.out) or len(files) > 1
                  or a.out.endswith(("/", "\\")) or not out_has_ext)
    if out_is_dir:
        os.makedirs(a.out, exist_ok=True)

    ver = entry["version"]
    guard = None if getattr(a, "no_clip_guard", False) else (0.97, 0.999)
    ref_key = (entry.get("key") or {}).get("src") or None
    match_mode = getattr(a, "match_key", "auto")
    results = []
    warnings = []
    non_display = {}
    low_bpp_srcs = []
    for f in files:
        img = load_rgb(f)
        info = {}
        person = None

        # Put the frame at the level the style was learned at. Without this a look that
        # lifts midtones blows out anything shot brighter than the reference pair.
        if match_mode == "off":
            info["key_match"] = {"applied": False, "reason": "disabled by --match-key off"}
        elif ref_key:
            in_key = image_key(img)
            stops = key_stops(in_key, ref_key)
            if match_mode == "on" or abs(stops) > 0.35:
                m_in = max(float(in_key["p50"]), 1e-4)
                m_ref = max(float(ref_key["p50"]), 1e-4)
                gamma = float(np.log(m_ref) / np.log(m_in)) if abs(np.log(m_in)) > 1e-6 else 1.0
                gamma = float(np.clip(gamma, 0.35, 3.0))
                matched = match_key(img, gamma)
                info["key_match"] = {
                    "applied": True,
                    "input_median": in_key["p50"],
                    "reference_median": m_ref,
                    "stops": round(stops, 2),
                    "gamma": round(gamma, 3),
                    "median_after": image_key(matched)["p50"],
                }
                img = matched
            else:
                info["key_match"] = {"applied": False, "stops": round(stops, 2),
                                     "reason": "within 0.35 stops of the reference"}
        else:
            info["key_match"] = {"applied": False,
                                 "reason": "this style version has no stored key; refit to enable"}

        if region_luts:
            # ask for "people" even when the style has no people LUT: the skin
            # check needs to know whether anyone is in frame before it judges hue
            req = list(region_luts) + ([] if "people" in region_luts else ["people"])
            m = segm.masks(img, regions=req)
            masks = {r: m.weight(r, img.shape[:2]) for r in region_luts}
            rendered = lc.apply_lut_regional(img, lut, region_luts, masks, clip_guard=guard)
            cov = m.coverage()
            person = m.person_pct(img)
            info["regions"] = {k: v for k, v in cov.items() if k in region_luts}
        else:
            rendered = lc.apply_lut(img, lut, clip_guard=guard)

        # reference-free check: what the style did to this frame, not what it was
        # supposed to do. A LUT that is right on the training pair can still crush
        # a night shot to black, and nothing in the fit metrics would notice.
        before = chk.measure(img, person_pct=person)
        after = chk.measure(rendered, person_pct=person)
        info["look"] = {
            "median_luma": [before["exposure"]["median_luma"], after["exposure"]["median_luma"]],
            "clipped_pct": [before["clipping"]["clipped_pct"], after["clipping"]["clipped_pct"]],
            "saturation": [before["mean_saturation"], after["mean_saturation"]],
        }
        for w in before["warnings"] + chk.compare(before, after):
            warnings.append({"file": os.path.basename(f), **w})
        tr = chk.transfer(img)
        if tr["verdict"] != "display":
            non_display[os.path.basename(f)] = tr

        if out_is_dir:
            ext = os.path.splitext(f)[1].lower()
            # keep the source container when we can write it; RAW/DNG cannot be written
            if ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
                ext = ".tif" if a.bits == 16 else ".png"
            elif a.bits == 16 and ext not in (".tif", ".tiff"):
                ext = ".tif"
            dest = os.path.join(a.out, f"{stem(f)}__{a.style}_v{ver}{ext}")
        else:
            dest = a.out
        meta = read_meta(f)
        mode = save_rgb(dest, rendered, bits=a.bits, quality=a.quality, meta=meta)
        rec = {"src": f, "out": dest, "mode": mode,
               "max_abs_pixel_change": round(float(np.abs(rendered - img).max()), 4),
               "icc_preserved": bool(meta.get("icc")), **info}

        # A lossless master alongside the delivery file. The JPEG is for sharing;
        # anything that will be edited again should come off this one.
        master = getattr(a, "master", "png")
        ext_src = os.path.splitext(dest)[1].lower()
        if master != "none" and ext_src not in (".png", ".tif", ".tiff", ".bmp"):
            m_ext = ".tif" if master == "tif" else ".png"
            m_bits = 16 if master == "tif" else 8
            m_path = os.path.splitext(dest)[0] + "__master" + m_ext
            try:
                rec["master"] = save_rgb(m_path, rendered, bits=m_bits, meta=meta)
                rec["master_path"] = m_path
            except SystemExit as exc:
                # a missing encoder must not throw away the delivery file that is
                # already on disk, nor the JSON the caller is waiting on
                m_path = os.path.splitext(dest)[0] + "__master.png"
                rec["master"] = save_rgb(m_path, rendered, bits=8, meta=meta)
                rec["master_path"] = m_path
                rec["master_note"] = f"{exc}; fell back to an 8-bit PNG master"
        results.append(rec)
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg"):
            # bytes per pixel is a blunt but reliable tell for a source that was
            # already squeezed by a chat app or a social platform before we saw it
            bpp = os.path.getsize(f) / max(img.shape[0] * img.shape[1], 1)
            if bpp < 0.10:
                low_bpp_srcs.append(os.path.basename(f))
    out = {"ok": True, "command": "apply", "style": a.style, "version": ver,
           "lut_space": ss.LUT_SPACE, "count": len(results), "files": results,
           "encoding": {"jpeg_quality": a.quality, "chroma": "4:4:4",
                        "master": getattr(a, "master", "png")}}
    if low_bpp_srcs:
        out["source_note"] = (
            f"这 {len(low_bpp_srcs)} 张源图本身已被强压缩（<0.10 字节/像素，社交软件转发的典型值），"
            "细节在进这一步之前就已经不在了，产物再无损也补不回来。"
            "要判断丢的是谁，请拿相机/原导出文件对照，不要拿被压过的副本。"
        )
    if warnings:
        out["warnings"] = warnings[:25]
        if len(warnings) > 25:
            out["warnings_truncated"] = len(warnings) - 25
    if non_display:
        out["input_transfer"] = non_display
    raw_inputs = [f for f in files if os.path.splitext(f)[1].lower() in RAW_EXT]
    if raw_inputs:
        out["raw_note"] = (
            f"{len(raw_inputs)} 张 RAW 由 libraw 近似解码（相机白平衡 + 通用色彩矩阵），"
            "未使用相机 DCP。与 Lightroom 的渲染可能有可见差异，需要精确交付请导 16bit TIFF。"
        )
    if auto_choice:
        out["auto"] = auto_choice
    if region_luts:
        out["regions_applied"] = sorted(region_luts)
    if region_note:
        out["region_note"] = region_note
    return out


def cmd_verify(a):
    ws = ss.resolve_workspace(a)
    lut, _meta = ss.load_lut(ws, a.style, a.version)
    region_luts, entry = ss.load_region_luts(ws, a.style, a.version)
    result = {"ok": True, "command": "verify", "style": a.style,
              "version": entry["version"], "health": lut_health(lut)}
    if region_luts:
        result["regions_checked"] = sorted(region_luts)
    if a.src and a.tgt:
        src, tgt = load_rgb(a.src), load_rgb(a.tgt)
        guard = None if getattr(a, "no_clip_guard", False) else (0.97, 0.999)
        if region_luts and segmod is not None:
            m = segmod.get_segmenter().masks(src, regions=list(region_luts))
            masks = {r: m.weight(r, src.shape[:2]) for r in region_luts}
            rendered = lc.apply_lut_regional(src, lut, region_luts, masks, clip_guard=guard)
        else:
            rendered = lc.apply_lut(src, lut, clip_guard=guard)
        result["fit_quality"] = color_metrics(rendered, tgt)
    if a.threshold is not None and "fit_quality" in result:
        de = result["fit_quality"].get("dE2000_mean")
        if isinstance(de, (int, float)):
            result["passes_threshold"] = bool(de <= a.threshold)
    return result


def cmd_check(a):
    """Describe a picture without a reference: what it is, and what is wrong."""
    files = iter_images(a.input)
    if not files:
        raise SystemExit(f"no images found at {a.input}")
    segm = None
    if a.person:
        if segmod is None:
            raise SystemExit("--person needs segmentation (onnxruntime + model)")
        segm = segmod.get_segmenter(quantized=a.seg_quantized)

    rows = []
    for f in files:
        img = load_rgb(f)
        person = None
        if segm is not None:
            m = segm.masks(img, regions=["people"])
            person = m.person_pct(img)
        meas = chk.measure(img, person_pct=person)
        warns = meas.pop("warnings", [])
        tr = chk.transfer(img)
        rows.append({"file": f, "transfer": tr, "measured": meas,
                     "warnings": warns,
                     "needs_attention": bool(warns) or tr["verdict"] == "log"})
    return {"ok": True, "command": "check", "count": len(rows),
            "any_attention": any(r["needs_attention"] for r in rows),
            "files": rows}


def cmd_styles(a):
    ws = ss.resolve_workspace(a)
    if a.name:
        man = ss.load_manifest(ws, a.name)
        return {"ok": True, "command": "styles", "style": a.name, "manifest": man}
    return {"ok": True, "command": "styles", "workspace": ws, "styles": ss.list_styles(ws)}


def cmd_export(a):
    """Write the LUT files (and, with --full, the node-wiring recipe).

    A regional style is not one LUT: it is a base transform plus one complete LUT per
    region, blended by a soft mask. Exporting just the base silently drops the regional
    work, so --full also writes the wiring notes that make the multi-node setup
    reproducible in another app.
    """
    ws = ss.resolve_workspace(a)
    lut, entry = ss.load_lut(ws, a.style, a.version)
    region_luts, entry2 = ss.load_region_luts(ws, a.style, a.version)
    ver = entry2["version"]
    strength = float(a.strength)
    if strength < 1.0 and strength > 0.0:
        lut = ss.blended_lut(lut, strength)

    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"{a.style}_v{ver}.cube"
    files = []
    p = os.path.join(out_dir, base_name)
    lc.write_cube(p, lut)
    files.append({"role": "base", "path": p, "size": lut.shape[0]})
    for r, rl in sorted(region_luts.items()):
        rp = os.path.join(out_dir, f"{a.style}_v{ver}_{r}.cube")
        lc.write_cube(rp, rl if strength >= 1.0 else ss.blended_lut(rl, strength))
        files.append({"role": f"region:{r}", "path": rp, "size": rl.shape[0]})

    note_path = None
    if a.full:
        note_path = os.path.join(out_dir, f"{a.style}_v{ver}_wiring.md")
        desc = entry2.get("description") or {}
        lines = [f"# {a.style} v{ver} — 节点接线说明", ""]
        if desc.get("summary"):
            lines += [f"风格：{desc['summary']}", ""]
        lines += [
            "## 为什么不是一个 .cube",
            "",
            "这个风格是分区的：先对整幅图做基础映射，再对特定区域用另一套映射覆盖。",
            "单个 3D LUT 是像素颜色的函数，无法表达\"同一颜色在不同区域不同处理\"，",
            "所以分区 LUT 只能拆成多个文件 + 掩膜。",
            "",
            "## 文件",
            "",
            f"- `{base_name}` — 基础层，整幅图都套用它。",
        ]
        for r in sorted(region_luts):
            lines.append(f"- `{a.style}_v{ver}_{r}.cube` — {r} 区域的**完整**LUT"
                         "（已包含基础层，可直接替换基础层，不是残差）。")
        lines += [
            "",
            "## 接线（Photoshop / Capture One / DaVinci / Affinity 通用）",
            "",
            "1. 图层/节点 1：对全图套用基础 LUT，强度 100%。",
            "2. 图层/节点 2 起：每个区域一个 LUT 图层，用该区域的**软掩膜**做图层蒙版，",
            "   蒙版灰度即混合权重（白=完全使用区域 LUT，黑=保持基础 LUT）。",
            "3. 掩膜要有 1~2% 画幅的羽化，硬边会在两种映射交界处出现轮廓线。",
            "",
            "渲染器内部等价于 `out = base*(1-m) + region*m`，所以蒙版值直接当不透明度用即可。",
            "",
            "## 需要注意",
            "",
            "- 区域 LUT 是**在基础 LUT 之上**拟合的残差再叠回基础，超出训练样本的颜色范围时",
            "  会退回基础层；这与\"区域 LUT 单独使用\"的结果不同，单独使用会丢掉基础层的整体风格。",
            "- 导出文件不含裁剪保护：本工具渲染时，接近过曝（最亮通道 ≥0.97）的像素会原样通过，",
            "  因为过曝像素没有颜色信息可映射。手工接线时需要自己在高光区减弱区域层，",
            "  否则逆光太阳一类光源会被压成灰块。",
        ]
        with open(note_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        files.append({"role": "notes", "path": note_path})

    return {"ok": True, "command": "export", "style": a.style, "version": ver,
            "out": out_dir, "files": files,
            "region_count": len(region_luts),
            "single_lut_possible": not region_luts,
            "note": ("分区风格无法合并为单个 .cube；按 wiring.md 接线"
                     if region_luts else "全局风格，单个 .cube 即可")}


def cmd_undo(a):
    """Remove one batch and refit from what is left. Adds a version, never deletes one."""
    ws = ss.resolve_workspace(a)
    before = len(ss.load_batches(ws, a.style))
    ss.remove_batch(ws, a.style, a.batch)

    man = ss.load_manifest(ws, a.style)
    cur = next((v for v in man["versions"] if v["version"] == man.get("current")), {})
    params = cur.get("params", {})
    regions = [r for r in (params.get("regions") or [])]
    size = int(cur.get("size") or 33)
    strength = float(params.get("region_strength") or 0.9)

    hist = ss.merge_batch_samples(ws, a.style, region_names=regions)
    if "S" not in hist or len(hist["S"]) == 0:
        # The batch is already gone at this point, so this cannot be reported as a
        # plain error and rolled back. Say exactly what happened, in JSON, so a
        # caller can tell "nothing changed" from "batch removed, LUT untouched".
        return {"ok": False, "command": "undo", "style": a.style,
                "removed_batch": a.batch, "batches_before": before, "batches_now": 0,
                "lut_changed": False,
                "error": "batch removed, but no archived samples remain; "
                         "the current LUT is unchanged"} 

    lut, diag = lc.fit_lut([(hist["S"], hist["T"])], size=size, samples_per_pair=a.samples)
    region_luts = {}
    for r in regions:
        if r in hist:
            rlut, _ = lc.fit_lut([hist[r]], size=size, samples_per_pair=a.samples,
                                 baseline=lut, residual_scale=strength, seed=1)
            region_luts[r] = rlut

    new_params = {**params, "mode": "fit", "undo_of": a.batch,
                  "batches_merged": len(ss.load_batches(ws, a.style))}
    # the key belongs to the look, not to the batch being dropped, so carry it over
    prev_key = cur.get("key")
    saved = ss.save_version(ws, a.style, lut, new_params,
                            {**diag, "undone_batch": a.batch}, region_luts=region_luts,
                            extra={"key": prev_key} if prev_key else None)
    return {"ok": True, "command": "undo", "style": a.style, "removed_batch": a.batch,
            "batches_before": before, "batches_now": len(ss.load_batches(ws, a.style)),
            **saved, "diagnostics": diag,
            "note": "refit from the remaining archived batches; earlier versions kept"}


def cmd_batches(a):
    ws = ss.resolve_workspace(a)
    bs = ss.load_batches(ws, a.style)
    return {"ok": True, "command": "batches", "style": a.style, "count": len(bs),
            "batches": bs}


# ---------------------------------------------------------------- parser


def build_parser():
    p = argparse.ArgumentParser(prog="grade", description="Deterministic AI colour grading (3D LUT).")
    p.add_argument("--workspace", default=None, help="style workspace dir (default: ./color-styles)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="learn a style from (original, retouched) pairs")
    f.add_argument("--pair", nargs=2, action="append", metavar=("SRC", "TGT"))
    f.add_argument("--src", help="dir of originals")
    f.add_argument("--tgt", help="dir of retouched images (matched by filename stem)")
    f.add_argument("--name", default=None,
                   help="existing name = merge into it; omit to auto-classify")
    f.add_argument("--auto-threshold", type=float, default=1.5,
                   help="ΔE below which a batch is folded into the nearest style")
    f.add_argument("--archive-samples", type=int, default=15000,
                   help="pixel pairs archived per image (for undo and re-anchor)")
    f.add_argument("--batch-id", default=None, help="override the generated batch id")
    f.add_argument("--size", type=int, default=33, choices=[17, 25, 33, 49, 65])
    f.add_argument("--samples", type=int, default=90000, help="samples per pair")
    f.add_argument("--smooth", type=int, default=2, help="smoothing passes")
    f.add_argument("--no-monotonic", dest="monotonic", action="store_false", default=True,
                   help="skip the per-axis monotonic pass (it is on by default; turning it "
                        "off can leave a couple of levels of tone inversion in deep shadow)")
    f.add_argument("--max-de", type=float, default=3.0, help="warn if training dE2000 exceeds this")
    f.add_argument("--regions", default="sky,people,veg",
                   help="comma list of regions to grade separately, empty = global only "
                        "(sky,people,veg,water,ground,arch)")
    f.add_argument("--region-strength", type=float, default=0.9,
                   help="how much of the regional residual to keep (0..1)")
    f.add_argument("--min-region-pixels", type=int, default=4000,
                   help="skip a region with fewer usable pixels than this")
    f.add_argument("--seg-quantized", action="store_true", default=True,
                   help="use the 4MB quantized segmentation model (default)")
    f.add_argument("--seg-full", dest="seg_quantized", action="store_false",
                   help="use the 15MB fp32 segmentation model")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("ref", help="derive a style from a reference image")
    r.add_argument("--ref", required=True)
    r.add_argument("--target", required=True, help="image the style is calibrated against")
    r.add_argument("--name", required=True)
    r.add_argument("--out", help="also render the target with the new style")
    r.add_argument("--size", type=int, default=33, choices=[17, 25, 33, 49, 65])
    r.add_argument("--strength", type=float, default=0.85)
    r.add_argument("--skin", type=float, default=0.55, help="skin protection factor (0..1)")
    r.add_argument("--bins", type=int, default=8, help="luminance bands")
    r.add_argument("--bits", type=int, default=8, choices=[8, 16])
    r.add_argument("--quality", type=int, default=98, help="JPEG quality when --out is a .jpg")
    r.set_defaults(func=cmd_ref)

    ap = sub.add_parser("apply", help="render images through a stored style")
    ap.add_argument("--style", required=True)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--in", dest="input", required=True, help="image file or directory")
    ap.add_argument("--out", required=True, help="output file or directory")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--bits", type=int, default=8, choices=[8, 16])
    ap.add_argument("--quality", type=int, default=98,
                   help="JPEG quality for the delivery file (default 98, always 4:4:4)")
    ap.add_argument("--master", default="png", choices=["png", "tif", "none"],
                   help="also write a lossless master next to the delivery file "
                        "(default png; tif = 16-bit; none = skip)")
    ap.add_argument("--seg-full", dest="seg_quantized", action="store_false", default=True,
                   help="use the fp32 segmentation model instead of the quantized one")
    ap.add_argument("--no-clip-guard", action="store_true",
                   help="apply the LUT to clipped pixels too (default: they pass through)")
    ap.add_argument("--match-key", dest="match_key", default="off",
                   choices=["auto", "on", "off"],
                   help="bring the frame to the exposure the style was learned at before "
                        "applying the LUT: off = apply as fitted (default), on = always, "
                        "auto = only when more than 0.35 stops off. Only safe when the "
                        "frame is lit like the reference: matching a frame average on "
                        "backlit material drags the subject under the background")
    ap.set_defaults(func=cmd_apply)

    v = sub.add_parser("verify", help="LUT self-check or fit quality check")
    v.add_argument("--style", required=True)
    v.add_argument("--version", type=int, default=None)
    v.add_argument("--src")
    v.add_argument("--tgt")
    v.add_argument("--threshold", type=float, default=None, help="fail if dE2000_mean exceeds")
    v.add_argument("--no-clip-guard", action="store_true")
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("check", help="reference-free check of one image or a folder")
    c.add_argument("--in", dest="input", required=True, help="image file or directory")
    c.add_argument("--person", action="store_true",
                   help="run segmentation so skin hue can be judged (needs a person in frame)")
    c.add_argument("--seg-full", dest="seg_quantized", action="store_false", default=True)
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("styles", help="list or inspect styles")
    s.add_argument("--name", default=None)
    s.set_defaults(func=cmd_styles)

    e = sub.add_parser("export", help="write the .cube files (and wiring notes)")
    e.add_argument("--style", required=True)
    e.add_argument("--version", type=int, default=None)
    e.add_argument("--out", required=True, help="output directory")
    e.add_argument("--strength", type=float, default=1.0)
    e.add_argument("--full", action="store_true",
                   help="also write the node-wiring notes for regional styles")
    e.set_defaults(func=cmd_export)

    u = sub.add_parser("undo", help="remove one batch and refit from the rest")
    u.add_argument("--style", required=True)
    u.add_argument("--batch", required=True)
    u.add_argument("--samples", type=int, default=90000)
    u.set_defaults(func=cmd_undo)

    b = sub.add_parser("batches", help="list the archived batches of a style")
    b.add_argument("--style", required=True)
    b.set_defaults(func=cmd_batches)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # surface as JSON so the agent can react
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("passes_threshold") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
