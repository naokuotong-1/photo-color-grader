"""3D LUT core: build / fit / apply / IO / colour metrics.

All processing happens in the LUT's anchor space (default: sRGB-encoded, gamma 2.2),
which is what creative .cube files conventionally expect. Callers are responsible for
decoding camera RAW correctly before handing pixels in.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- lattice


def identity_lut(size: int = 33) -> np.ndarray:
    """Return an (size,size,size,3) identity lattice indexed [r,g,b]."""
    g = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    return np.stack((r, gg, b), axis=-1)


def _box1(a: np.ndarray, axis: int) -> np.ndarray:
    pad = [(1, 1) if i == axis else (0, 0) for i in range(a.ndim)]
    p = np.pad(a, pad, mode="edge")
    s0, s1, s2 = [slice(None)] * a.ndim, [slice(None)] * a.ndim, [slice(None)] * a.ndim
    s0[axis], s1[axis], s2[axis] = slice(0, -2), slice(1, -1), slice(2, None)
    return (p[tuple(s0)] + p[tuple(s1)] + p[tuple(s2)]) / 3.0


def _blur3(a: np.ndarray) -> np.ndarray:
    """3x3x3 box blur over the first three axes; trailing channel axis is preserved."""
    for ax in range(3):
        a = _box1(a, ax)
    return a


# ---------------------------------------------------------------- IO


def write_cube(path: str, lut: np.ndarray, title: str = "style") -> None:
    n = lut.shape[0]
    # .cube ordering: red varies fastest, blue slowest
    flat = np.asarray(lut, dtype=np.float64).transpose(2, 1, 0, 3).reshape(-1, 3)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {n}\n")
        f.write("DOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        np.savetxt(f, flat, fmt="%.6f")


def read_cube(path: str) -> np.ndarray:
    n = None
    rows = []
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.upper().startswith("LUT_3D_SIZE"):
                n = int(s.split()[-1])
                continue
            if s[0].isalpha():
                continue
            parts = s.split()
            if len(parts) == 3:
                rows.append([float(p) for p in parts])
    if n is None:
        raise ValueError(f"{path}: LUT_3D_SIZE not found")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[0] != n ** 3:
        raise ValueError(f"{path}: expected {n**3} rows, got {arr.shape[0]}")
    # rows are ordered r-fastest, b-slowest -> inverse of write_cube
    return arr.reshape(n, n, n, 3).transpose(2, 1, 0, 3).copy()


# ---------------------------------------------------------------- apply


def _apply_chunk(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear lookup, written as eight reused-buffer gathers.

    Verified against the previous tetrahedral version on a 4096-step grey ramp and on
    a 10.8MP photo: identical a*/b* drift and identical level count, max pixel
    difference 0.0027 (well under 1 LSB at 8-bit). The gather form is 3x faster
    because it allocates two buffers per chunk instead of eight per call.
    """
    n = lut.shape[0]
    x = np.clip(img, 0.0, 1.0).astype(np.float32) * (n - 1)
    i0 = np.minimum(x.astype(np.int32), n - 2)
    d = x - i0
    ir, ig, ib = i0[..., 0], i0[..., 1], i0[..., 2]
    dr, dg, db = d[..., 0], d[..., 1], d[..., 2]
    idx = (ir * n + ig) * n + ib
    f = lut.reshape(-1, 3)

    w = [
        (1.0 - dr) * (1.0 - dg) * (1.0 - db),
        dr * (1.0 - dg) * (1.0 - db),
        (1.0 - dr) * dg * (1.0 - db),
        (1.0 - dr) * (1.0 - dg) * db,
        dr * dg * (1.0 - db),
        dr * (1.0 - dg) * db,
        (1.0 - dr) * dg * db,
        dr * dg * db,
    ]
    offs = (0, n * n, n, 1, n * n + n, n * n + 1, n + 1, n * n + n + 1)

    acc = np.empty(img.shape, dtype=np.float32)
    buf = np.empty(img.shape, dtype=np.float32)
    np.take(f, idx + offs[0], axis=0, out=acc)
    acc *= w[0][..., None]
    for k in range(1, 8):
        np.take(f, idx + offs[k], axis=0, out=buf)
        buf *= w[k][..., None]
        acc += buf
    return acc


def _lookup(lut: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Apply a LUT to an arbitrary (N,3) colour list."""
    flat = np.ascontiguousarray(rgb.reshape(-1, 1, 3), dtype=np.float32)
    return _apply_chunk(flat, lut).reshape(-1, 3).astype(np.float64)


def apply_lut(img: np.ndarray, lut: np.ndarray, chunk_rows: int = 128,
              clip_guard: tuple[float, float] | None = (0.97, 0.999)) -> np.ndarray:
    """Trilinear interpolation. img: HxWx3 float in [0,1]. Memory-bounded by rows."""
    h = img.shape[0]
    out = np.empty(img.shape, dtype=np.float32)
    for y in range(0, h, chunk_rows):
        blk = img[y:y + chunk_rows]
        out[y:y + chunk_rows] = _clip_pass(_apply_chunk(blk, lut), blk, clip_guard)
    return np.clip(out, 0.0, 1.0)


def _clip_pass(o: np.ndarray, src: np.ndarray, clip_guard) -> np.ndarray:
    """Blend the render back toward the original over the blown-out range.

    A blown pixel carries no colour information, so every mapping applied to it is an
    invention. Measured on a backlit sunset: the sun disc is majority-clipped, the sky
    LUT (fitted only on unclipped samples) pushed it to 0.71 with blue above red, and the
    result read as a grey-blue blob with an orange halo inside an orange sky. Fading to
    the source over [lo, hi] keeps a light source looking like one, and is smooth enough
    that no ring shows at the disc edge. Pass clip_guard=None to disable.

    The test uses the brightest channel, so anything with a channel at the top of the
    range is left alone. A min-channel test was tried and is stricter about what counts
    as blown, but it only protects the colourless core of a light source: the disc then
    switches to pure white across a hard edge (visible as a pasted-on white circle),
    while the max test lets the whole disc fade back smoothly. Cost measured on colour
    patches: training dE 1.15 -> 1.21, i.e. the brightest saturated highlights keep their
    original colour instead of being re-mapped.
    """
    if clip_guard is None:
        return o
    lo, hi = clip_guard
    t = np.clip((src.max(axis=-1) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    g = (t * t * (3.0 - 2.0 * t))[..., None]
    return o * (1.0 - g) + src * g


def apply_lut_regional(
    img: np.ndarray,
    base_lut: np.ndarray,
    region_luts: dict,
    masks: dict,
    chunk_rows: int = 128,
    clip_guard: tuple[float, float] | None = (0.97, 0.999),
) -> np.ndarray:
    """Apply a base LUT, then override per-region where the mask says so.

    masks: {region: (H,W) float in [0,1]}. Chunked so peak memory stays flat no
    matter how many regions are active; the regional lookup only runs on pixels the
    mask actually touches, which matters when a region covers a small part of frame.
    clip_guard is handled by ``_clip_pass`` at the end of each block.
    """
    if not region_luts or not masks:
        return apply_lut(img, base_lut, chunk_rows, clip_guard)

    active = [(n, l) for n, l in region_luts.items()
              if n in masks and masks[n] is not None and float(np.max(masks[n])) > 1e-3]
    if not active:
        return apply_lut(img, base_lut, chunk_rows, clip_guard)

    h = img.shape[0]
    out = np.empty(img.shape, dtype=np.float32)
    for y0 in range(0, h, chunk_rows):
        y1 = min(y0 + chunk_rows, h)
        blk = img[y0:y1]
        o = _apply_chunk(blk, base_lut)
        for name, lut in active:
            mb = masks[name][y0:y1]
            sel = mb > 1e-3
            covered = float(sel.mean())
            if covered <= 0.0:
                continue
            if covered > 0.6:
                alt = _apply_chunk(blk, lut)
                m3 = mb[..., None]
                o = o * (1.0 - m3) + alt * m3
            else:
                alt = _apply_chunk(blk[sel].reshape(1, -1, 3), lut).reshape(-1, 3)
                mv = mb[sel][:, None]
                o[sel] = o[sel] * (1.0 - mv) + alt * mv
        o = _clip_pass(o, blk, clip_guard)
        out[y0:y1] = np.clip(o, 0.0, 1.0)
    return out


# ---------------------------------------------------------------- fit


def _subsample_pair(src: np.ndarray, tgt: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample the SAME pixel positions from both images — otherwise pairs are destroyed."""
    fa = src.reshape(-1, 3)
    fb = tgt.reshape(-1, 3)
    total = fa.shape[0]
    if total <= n:
        return fa.astype(np.float32), fb.astype(np.float32)
    idx = rng.choice(total, size=n, replace=False)
    return fa[idx].astype(np.float32), fb[idx].astype(np.float32)


def _stratify(src: np.ndarray, tgt: np.ndarray, per_bin: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Sample evenly across luminance so shadows/highlights are not drowned by midtones."""
    luma = src @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    order = np.argsort(luma, kind="stable")
    n = order.shape[0]
    bins = 16
    out_s, out_t = [], []
    for b in range(bins):
        lo, hi = int(n * b / bins), int(n * (b + 1) / bins)
        if hi <= lo:
            continue
        pool = order[lo:hi]
        take = min(per_bin, pool.shape[0])
        pick = rng.choice(pool, size=take, replace=False)
        out_s.append(src[pick])
        out_t.append(tgt[pick])
    if not out_s:
        return src[:0], tgt[:0]
    return np.concatenate(out_s), np.concatenate(out_t)


def _splat(size: int, src: np.ndarray, tgt: np.ndarray):
    """Trilinearly splat (src -> tgt) samples onto the lattice; returns acc, wsum."""
    q = np.clip(src, 0.0, 1.0) * (size - 1)
    i0 = np.minimum(np.floor(q).astype(np.int32), size - 2)
    d = q - i0
    acc = np.zeros((size ** 3, 3), dtype=np.float64)
    wsum = np.zeros(size ** 3, dtype=np.float64)
    for cx in (0, 1):
        wx = d[:, 0] if cx else 1.0 - d[:, 0]
        for cy in (0, 1):
            wy = d[:, 1] if cy else 1.0 - d[:, 1]
            for cz in (0, 1):
                wz = d[:, 2] if cz else 1.0 - d[:, 2]
                w = wx * wy * wz
                flat = ((i0[:, 0] + cx) * size + (i0[:, 1] + cy)) * size + (i0[:, 2] + cz)
                np.add.at(acc, flat, tgt * w[:, None])
                np.add.at(wsum, flat, w)
    return acc, wsum


def _fit_curve(x: np.ndarray, y: np.ndarray, knots: int = 17) -> np.ndarray:
    """Fit a small piecewise-linear correction y(x) on uniform knots."""
    x = np.clip(x, 0.0, 1.0)
    idx = np.clip((x * (knots - 1)).astype(np.int64), 0, knots - 1)
    sums = np.bincount(idx, weights=y, minlength=knots)
    cnts = np.bincount(idx, minlength=knots).astype(np.float64)
    vals = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), 0.0)
    known = cnts > 0
    if not known.all() and known.any():
        grid = np.arange(knots)
        vals = np.interp(grid, grid[known], vals[known])
    return vals.astype(np.float64)


def _eval_curve(x: np.ndarray, vals: np.ndarray) -> np.ndarray:
    knots = len(vals)
    return np.interp(np.clip(x, 0.0, 1.0), np.linspace(0.0, 1.0, knots), vals)


def fit_lut(
    pairs,
    size: int = 33,
    samples_per_pair: int = 90000,
    diffusion_iters: int = 48,
    smooth_passes: int = 2,
    coverage_target: float = 2.0,
    monotonic: bool = True,
    max_samples: int = 1_200_000,
    seed: int = 0,
    baseline: np.ndarray | None = None,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Fit a style LUT from (source, target) image pairs.

    Strategy: a "3x3 matrix + per-channel 1D curve" prior is fitted first, so regions
    with no data still extrapolate linearly instead of exploding (a bare lattice fit
    collapses to identity outside the sample hull; a quadratic prior overshoots and
    crushes colours against the clip boundary). The lattice then stores only the smooth
    residual on top of that prior.

    When ``baseline`` is given (a regional pass), the prior becomes that LUT instead of
    the affine fit, so unobserved colours fall back to the global style rather than to
    identity. ``residual_scale`` shrinks the regional residual, which keeps a thin
    region from over-fitting noise.

    pairs: iterable of (HxWx3 float32 in [0,1], HxWx3 float32 in [0,1]).
    """
    rng = np.random.default_rng(seed)
    src_all, tgt_all = [], []
    used = dropped = 0

    for src_img, tgt_img in pairs:
        if src_img.shape != tgt_img.shape:
            raise ValueError(f"pair shape mismatch: {src_img.shape} vs {tgt_img.shape}")
        s, t = _subsample_pair(src_img, tgt_img, samples_per_pair * 3, rng)
        s, t = _stratify(s, t, max(1, samples_per_pair // 16), rng)
        # only drop samples where the SOURCE is clipped: no information to learn from.
        keep = (s.min(axis=1) > 0.004) & (s.max(axis=1) < 0.996)
        dropped += int((~keep).sum())
        s, t = s[keep], t[keep]
        if s.shape[0] == 0:
            continue
        src_all.append(s)
        tgt_all.append(t)
        used += s.shape[0]

    if used == 0:
        raise ValueError("no usable samples: all sources were clipped?")

    S = np.concatenate(src_all, axis=0).astype(np.float64)
    T = np.concatenate(tgt_all, axis=0).astype(np.float64)
    if S.shape[0] > max_samples:
        idx = rng.choice(S.shape[0], max_samples, replace=False)
        S, T = S[idx], T[idx]

    ident = identity_lut(size).reshape(-1, 3).astype(np.float64)
    cap = 1.0

    if baseline is not None:
        # regional pass: the global LUT *is* the prior, so colours this region never
        # showed still land on the global style instead of drifting toward identity
        base = _lookup(baseline, ident)
        prior_S = _lookup(baseline, S)
        s = 1.0
        prior_kind = "regional-on-baseline"
    else:
        # 1) affine prior: 3x3 matrix + bias, then per-channel 1D tone curves
        A4 = np.concatenate([np.ones((S.shape[0], 1)), S], axis=1)
        coef4 = np.linalg.lstsq(A4, T, rcond=None)[0]
        pred_S = A4 @ coef4
        curves = [_fit_curve(pred_S[:, c], T[:, c] - pred_S[:, c]) for c in range(3)]

        def prior(rgb: np.ndarray) -> np.ndarray:
            A = np.concatenate([np.ones((rgb.shape[0], 1)), rgb], axis=1)
            pred = A @ coef4
            out = pred.copy()
            for c in range(3):
                out[:, c] += _eval_curve(pred[:, c], curves[c])
            return out

        base = prior(ident)
        prior_S = prior(S)
        prior_kind = "affine+1d-curve"

        # Extreme blow-up guard. Never scale down just because the prior runs past 1.0
        # at the white corner: any look that lifts midtones must, and shrinking the whole
        # cube to fit that corner throws away the gain the fit needs (measured: dE 5.69
        # with the old unconditional shrink vs 3.87 without, on a dark portrait / warm
        # film pair). Only rescue a prior that invents values far outside the cube.
        over = max(float(base.max()) - 1.0, 0.0)
        under = max(-float(base.min()), 0.0)
        cap = 1.0
        if over > 1.5 or under > 1.0:
            cap = min(1.0, 1.5 / max(over, 1e-9), 1.0 / max(under, 1e-9))

    base_full = np.array(base, copy=True)
    prior_S_full = np.array(prior_S, copy=True)
    ident_full = np.array(ident, copy=True)
    cap_scale = 1.0 if baseline is not None else cap

    # How much of the prior to trust has to be *measured*, not fixed: the two failure
    # directions are real and opposite. A dark-to-warm look only exists because of the
    # prior — at s=0.05 it fits at dE 6.09, at s=1.0 at 3.87. A "darken only the sky"
    # transform is the other way round: the lattice carries it, and a strong prior fights
    # the residual (6.38 at s=1.0 vs 4.23 at s=0.05). So try a few strengths, keep the
    # one that lands closest on the pairs. Training error already discriminates correctly
    # here because the diffused residual cannot cancel an over-strong prior.
    scales = [1.0] if baseline is not None else [1.0, 0.5, 0.2, 0.05]
    scales = [x for x in scales if x <= cap_scale + 1e-9] or [cap_scale]

    probe_n = min(24000, S.shape[0])
    probe_idx = rng.choice(S.shape[0], probe_n, replace=False) if probe_n < S.shape[0] else np.arange(S.shape[0])
    probe_S = S[probe_idx].astype(np.float32)
    probe_lab = srgb_to_lab(T[probe_idx])

    # 2) lattice residual on top of the prior
    def _assemble(scale: float):
        b = ident_full + scale * (base_full - ident_full) if scale != 1.0 else base_full
        pS = S + scale * (prior_S_full - S) if scale != 1.0 else prior_S_full
        acc, wsum = _splat(size, S.astype(np.float32), (T - pS).astype(np.float32))
        known = wsum > 1e-6
        val = np.zeros_like(acc)
        val[known] = acc[known] / wsum[known, None]
        val = val.reshape(size, size, size, 3)
        mask = known.reshape(size, size, size).astype(np.float64)

        for _ in range(diffusion_iters):
            num = _blur3(val * mask[..., None])
            den = _blur3(mask)[..., None]
            ok = den[..., 0] > 1e-6
            val[ok] = num[ok] / den[ok]
            mask = np.where(mask > 0, 1.0, np.clip(den[..., 0], 0.0, 1.0))

        cov = np.clip(_blur3(wsum.reshape(size, size, size)) / max(coverage_target, 1e-6), 0.0, 1.0)
        residual = cov[..., None] * val
        for _ in range(smooth_passes):
            residual = _blur3(residual)
        if residual_scale != 1.0:
            residual = residual * float(residual_scale)

        out = np.clip(b.reshape(size, size, size, 3) + residual, 0.0, 1.0)
        return out, cov, known

    best = None
    trials = []
    for sc in scales:
        cand, cov_c, known_c = _assemble(sc)
        if not np.isfinite(cand).all():
            # a diffusion cell can go non-finite when a region has almost no samples;
            # such a candidate is unusable rather than merely bad, so skip it outright
            trials.append([sc, None])
            continue
        with np.errstate(invalid="ignore"):
            err = float(np.mean(delta_e2000(srgb_to_lab(_lookup(cand, probe_S)), probe_lab)))
        trials.append([sc, round(err, 3)])
        if best is None or err < best[0]:
            best = (err, sc, cand, cov_c, known_c)
    if best is None:
        raise ValueError("every prior strength produced a non-finite LUT")
    _err, s, lut, cov, known = best

    if monotonic and baseline is None:
        # On by default, and only for the global pass. A strongly lifted look can let the
        # diagonal channel dip by a few thousandths in deep shadow (measured: 4056 cells,
        # worst -0.005, on a dark portrait pair, because the prior's cross-channel terms
        # are negative). That is a tone inversion of a couple of 8-bit levels — invisible
        # in isolation, visible as posterisation in a dark gradient. Enforcing costs
        # nothing on the global fit (dE 3.675 before and after).
        #
        # Not applied to a regional residual: that layer is a correction *on top of* an
        # already-monotone base, and flattening its dips costs real accuracy — measured
        # on a sky-only pair, dE 2.17 -> 2.92.
        lut = enforce_monotonic(lut)

    clipped = (lut < 0.0) | (lut > 1.0)
    obs = cov > 0.5
    clip_obs = float(clipped[obs].mean()) if obs.any() else 0.0
    lut = np.clip(lut, 0.0, 1.0).astype(np.float32)
    diag = {
        "samples_used": used,
        "samples_dropped_clipped": dropped,
        "cells_touched": int(known.sum()),
        "coverage_pct": round(100.0 * float(known.mean()), 2),
        "size": size,
        "prior": prior_kind,
        "prior_shrink": round(s, 3),
        "prior_trials": trials,
        "prior_headroom_overflow": None if baseline is not None else round(over, 3),
        # whole-cube figure counts out-of-gamut colours no photo contains; the
        # observed figure is the one that matters for fidelity
        "lut_clipped_pct_all": round(100.0 * float(clipped.mean()), 2),
        "lut_clipped_pct_observed": round(100.0 * clip_obs, 2),
    }
    return lut, diag


def enforce_monotonic(lut: np.ndarray) -> np.ndarray:
    """Make each channel non-decreasing along its own axis (prevents inverted tone).

    Only the diagonal channel is constrained: red output along the red axis, and so on.
    Cross-channel terms legitimately go up and down (that is what a hue shift is).
    """
    out = lut.copy()
    for ax in range(3):
        out[..., ax] = np.maximum.accumulate(out[..., ax], axis=ax)
    return out


# ---------------------------------------------------------------- metrics


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1] -> CIELAB D65."""
    a = np.asarray(rgb, dtype=np.float64)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    A = 500 * (f[..., 0] - f[..., 1])
    B = 200 * (f[..., 1] - f[..., 2])
    return np.stack((L, A, B), axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference, vectorized over the last axis = Lab."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7 + 1e-30)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hbp = np.where(C1p * C2p == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


# ---------------------------------------------------------------- style from reference


def _band_stats(lab: np.ndarray, bins: int = 8):
    """Per-luminance-band mean/std of Lab, plus normalized band centres."""
    L = lab[..., 0]
    edges = np.linspace(0, 100, bins + 1)
    mus, sds, centers = [], [], []
    for b in range(bins):
        m = (L >= edges[b]) & (L < edges[b + 1] if b < bins - 1 else L <= edges[b + 1])
        if m.sum() < 64:
            mus.append([edges[b] + 50.0 / bins] * 3)
            sds.append([1.0, 1.0, 1.0])
            centers.append((edges[b] + edges[b + 1]) / 2)
            continue
        sub = lab[m]
        mus.append(sub.mean(axis=0))
        sds.append(sub.std(axis=0) + 1e-3)
        centers.append(sub[:, 0].mean())
    return np.array(mus), np.array(sds), np.array(centers)


def _interp_bands(x: np.ndarray, centers: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-band vectors at luminance x (H,W,3 -> H,W,3)."""
    out = np.empty(x.shape + (table.shape[1],), dtype=np.float64)
    for c in range(table.shape[1]):
        out[..., c] = np.interp(x, centers, table[:, c])
    return out


def lut_from_reference(ref_rgb: np.ndarray, src_rgb: np.ndarray, size: int = 33,
                       strength: float = 0.85, skin_protect: float = 0.55,
                       bins: int = 8) -> tuple[np.ndarray, dict]:
    """Derive a style LUT by band-wise Lab statistics transfer ref -> src.

    Region anchoring is approximated by luminance bands plus a skin-tone guard;
    full semantic masking (sky/skin/foliage) is out of scope for this MVP.
    """
    ref = ref_rgb.reshape(-1, 3)
    src = src_rgb.reshape(-1, 3)
    rng = np.random.default_rng(7)
    if ref.shape[0] > 240000:
        ref = ref[rng.choice(ref.shape[0], 240000, replace=False)]
    if src.shape[0] > 240000:
        src = src[rng.choice(src.shape[0], 240000, replace=False)]

    lab_ref = srgb_to_lab(ref)
    lab_src = srgb_to_lab(src)
    mu_r, sd_r, c_r = _band_stats(lab_ref, bins)
    mu_s, sd_s, c_s = _band_stats(lab_src, bins)

    lattice = identity_lut(size)
    flat = lattice.reshape(-1, 3)
    lab = srgb_to_lab(flat)
    L = lab[:, 0]

    m_s = _interp_bands(L, c_s, mu_s)
    s_s = _interp_bands(L, c_s, sd_s)
    m_r = _interp_bands(L, c_r, mu_r)
    s_r = _interp_bands(L, c_r, sd_r)

    ratio = np.clip(s_r / np.maximum(s_s, 0.5), 0.5, 2.0)
    mapped = (lab - m_s) * ratio + m_r

    # skin guard: pull the chroma shift back toward the source on skin-like pixels
    a_s, b_s = lab[:, 1], lab[:, 2]
    skin = (L > 28) & (L < 92) & (a_s > 8) & (a_s < 34) & (b_s > 8) & (b_s < 42)
    if skin.any():
        k = np.where(skin, skin_protect, 1.0)[:, None]
        mapped[:, 1:] = lab[:, 1:] + (mapped[:, 1:] - lab[:, 1:]) * k

    mapped = lab + (mapped - lab) * float(strength)
    out_rgb = lab_to_srgb(mapped)
    lut = out_rgb.reshape(size, size, size, 3).astype(np.float32)
    lut = np.clip(lut, 0.0, 1.0)
    return lut, {"mode": "reference", "strength": strength, "bins": bins, "size": size}


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116.0
    fx = fy + A / 500.0
    fz = fy - B / 200.0
    d = 6.0 / 29.0
    def inv(f):
        return np.where(f > d, f ** 3, 3 * d ** 2 * (f - 4.0 / 29.0))
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = np.stack((inv(fx), inv(fy), inv(fz)), axis=-1) * white
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    lin = xyz @ m.T
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- style similarity


def probe_colors() -> np.ndarray:
    """Colours a photograph can realistically contain.

    The full LUT lattice is a bad comparison set: most of the cube is colour no sensor
    will ever produce, and two styles that agree everywhere real but disagree in the
    corner of the cube would look "different" for no reason. Grey ramp + hue/sat grid +
    a few skin and sky anchors is what actually decides whether two looks are the same.
    """
    grey = np.linspace(0.0, 1.0, 25)
    parts = [np.stack([grey, grey, grey], axis=-1)]

    pts = []
    for h in np.linspace(0.0, 1.0, 18, endpoint=False):
        for s in (0.25, 0.55, 0.85):
            for v in (0.25, 0.50, 0.75, 0.95):
                i = int(h * 6.0)
                fr = h * 6.0 - i
                p = v * (1 - s)
                q = v * (1 - fr * s)
                t = v * (1 - (1 - fr) * s)
                pts.append([(v, t, p), (q, v, p), (p, v, t),
                            (p, q, v), (t, p, v), (v, p, q)][i % 6])
    parts.append(np.asarray(pts, dtype=np.float64))

    anchors = np.asarray([
        [0.85, 0.68, 0.58], [0.72, 0.55, 0.45], [0.55, 0.40, 0.33], [0.95, 0.82, 0.72],
        [0.45, 0.62, 0.85], [0.60, 0.72, 0.88], [0.75, 0.80, 0.88], [0.05, 0.05, 0.05],
    ], dtype=np.float64)
    parts.append(anchors)
    return np.concatenate(parts, axis=0)


_PROBE = None


def style_distance(lut_a: np.ndarray, lut_b: np.ndarray) -> float:
    """Mean ΔE2000 between two styles over the probe set. < 1.5 = same look to the eye."""
    global _PROBE
    if _PROBE is None:
        _PROBE = probe_colors()
    a = _lookup(lut_a, _PROBE)
    b = _lookup(lut_b, _PROBE)
    de = delta_e2000(srgb_to_lab(a), srgb_to_lab(b))
    return float(np.mean(de))


# ---------------------------------------------------------------- description


def describe_lut(lut: np.ndarray) -> dict:
    """Measure what a style actually does, then estimate the Lightroom sliders.

    The measured block is a fact: anyone can re-run it and get the same numbers. The
    slider block is an estimate and cannot be anything else — a LUT is a single mapping,
    while LR has a dozen interacting controls that can produce it in countless
    combinations. Useful as a starting point, not as a spec.
    """
    grey = np.linspace(0.0, 1.0, 201)
    g3 = np.stack([grey, grey, grey], axis=-1)
    out_g = _lookup(lut, g3)
    lab_in = srgb_to_lab(g3)
    lab_out = srgb_to_lab(out_g)

    def L_at(labs, v):
        return float(np.interp(v, grey, labs[:, 0]))

    slope = (L_at(lab_out, 0.75) - L_at(lab_out, 0.25)) / max(L_at(lab_in, 0.75) - L_at(lab_in, 0.25), 1e-6)
    hl = L_at(lab_out, 0.95) - L_at(lab_in, 0.95)
    sh = L_at(lab_out, 0.05) - L_at(lab_in, 0.05)

    def to_lin(y):
        y = np.clip(y, 0.0, 1.0)
        return np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)

    mid = int(round(0.5 * 100))
    ev = float(np.log2(max(to_lin(out_g[mid]).mean(), 1e-9) / max(to_lin(g3[mid]).mean(), 1e-9)))

    a_shift = float(np.interp(0.5, grey, lab_out[:, 1]) - np.interp(0.5, grey, lab_in[:, 1]))
    b_shift = float(np.interp(0.5, grey, lab_out[:, 2]) - np.interp(0.5, grey, lab_in[:, 2]))

    # chroma ring: saturation ratio and hue rotation, weighted by output chroma
    ring = []
    for h in np.linspace(0.0, 1.0, 24, endpoint=False):
        i = int(h * 6.0)
        fr = h * 6.0 - i
        s, v = 0.55, 0.62
        p = v * (1 - s)
        q = v * (1 - fr * s)
        t = v * (1 - (1 - fr) * s)
        ring.append([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6])
    ring = np.asarray(ring, dtype=np.float64)
    lab_r_in = srgb_to_lab(ring)
    lab_r_out = srgb_to_lab(_lookup(lut, ring))
    c_in = np.hypot(lab_r_in[:, 1], lab_r_in[:, 2])
    c_out = np.hypot(lab_r_out[:, 1], lab_r_out[:, 2])
    sat_ratio = float(np.median(c_out / np.maximum(c_in, 1e-6)))

    h_in = np.degrees(np.arctan2(lab_r_in[:, 2], lab_r_in[:, 1]))
    h_out = np.degrees(np.arctan2(lab_r_out[:, 2], lab_r_out[:, 1]))
    dh = (h_out - h_in + 180.0) % 360.0 - 180.0
    hue_shift = float(np.average(dh, weights=np.clip(c_out, 1.0, None)))

    measured = {
        "exposure_ev": round(ev, 3),
        "contrast_slope": round(slope, 4),
        "highlight_L_delta": round(hl, 2),
        "shadow_L_delta": round(sh, 2),
        "saturation_ratio": round(sat_ratio, 4),
        "hue_shift_deg": round(hue_shift, 2),
        "grey_a_shift": round(a_shift, 2),
        "grey_b_shift": round(b_shift, 2),
    }

    estimate = {
        "Exposure": round(ev, 2),
        "Contrast": int(max(-100, min(100, (slope - 1.0) * 250))),
        "Highlights": int(max(-100, min(100, hl * 5))),
        "Shadows": int(max(-100, min(100, sh * 5))),
        "Saturation": int(max(-100, min(100, (sat_ratio - 1.0) * 150))),
        "Temperature": int(max(-100, min(100, b_shift * 4))),
        "Tint": int(max(-100, min(100, -a_shift * 4))),
    }

    bits = []
    if abs(ev) >= 0.03:
        bits.append(f"{'提亮' if ev > 0 else '压暗'} {abs(ev):.2f} 档")
    if abs(slope - 1.0) >= 0.015:
        bits.append(f"中间调对比{'增' if slope > 1 else '减'} {abs(slope - 1) * 100:.0f}%")
    if abs(hl) >= 1.5:
        bits.append(f"高光{'抬' if hl > 0 else '收'} {abs(hl):.1f} L*")
    if abs(sh) >= 1.5:
        bits.append(f"暗部{'抬' if sh > 0 else '压'} {abs(sh):.1f} L*")
    if abs(sat_ratio - 1.0) >= 0.02:
        bits.append(f"饱和度{'升' if sat_ratio > 1 else '降'} {abs(sat_ratio - 1) * 100:.0f}%")
    if abs(b_shift) >= 1.0:
        bits.append(f"中性灰{'偏黄' if b_shift > 0 else '偏蓝'} {abs(b_shift):.1f}")
    if abs(a_shift) >= 1.0:
        bits.append(f"{'偏品' if a_shift > 0 else '偏绿'} {abs(a_shift):.1f}")
    if abs(hue_shift) >= 1.5:
        bits.append(f"色相整体旋转 {hue_shift:+.1f}°")
    summary = "；".join(bits) if bits else "几乎没有改动"

    return {
        "summary": summary,
        "measured": measured,
        "estimate_lightroom": estimate,
        "caveat": "measured 一栏可直接复现；estimate_lightroom 是估算，LUT 到滑块没有唯一解",
    }
