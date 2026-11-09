"""Versioned style store: color-styles/<name>/style.json + vN.cube"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

import lut_core as lc

DEFAULT_WORKSPACE = "color-styles"
LUT_SPACE = "srgb-gamma-2.2"


def resolve_workspace(args) -> str:
    ws = getattr(args, "workspace", None) or DEFAULT_WORKSPACE
    os.makedirs(ws, exist_ok=True)
    return ws


def _dir(ws: str, name: str) -> str:
    d = os.path.join(ws, name)
    os.makedirs(d, exist_ok=True)
    return d


def _manifest_path(ws: str, name: str) -> str:
    return os.path.join(_dir(ws, name), "style.json")


def load_manifest(ws: str, name: str) -> dict:
    p = _manifest_path(ws, name)
    if not os.path.exists(p):
        raise SystemExit(f"style not found: {name} (looked in {p})")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_version(ws: str, name: str, lut: np.ndarray, params: dict, metrics: dict,
                 region_luts: dict | None = None, extra: dict | None = None) -> dict:
    d = _dir(ws, name)
    p = _manifest_path(ws, name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            man = json.load(f)
    else:
        man = {"style_id": name, "lut_space": LUT_SPACE, "versions": []}

    version = len(man["versions"]) + 1
    fname = f"v{version}.cube"
    lc.write_cube(os.path.join(d, fname), lut, title=f"{name} v{version}")

    regions = {}
    for region, rlut in (region_luts or {}).items():
        rfname = f"v{version}_{region}.cube"
        lc.write_cube(os.path.join(d, rfname), rlut, title=f"{name} v{version} {region}")
        regions[region] = rfname

    entry = {
        "version": version,
        "cube": fname,
        "size": int(lut.shape[0]),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "metrics": metrics,
    }
    if regions:
        entry["regions"] = regions
    if extra:
        entry.update(extra)
    man["versions"].append(entry)
    man["current"] = version
    with open(p, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    out = {"style_id": name, "version": version, "cube": os.path.join(d, fname)}
    if regions:
        out["region_cubes"] = {k: os.path.join(d, v) for k, v in regions.items()}
    return out


def load_region_luts(ws: str, name: str, version: int | None = None) -> tuple[dict, dict]:
    """Return ({region: lut}, entry) for a version; empty dict when it has no regions."""
    man = load_manifest(ws, name)
    v = version or man.get("current")
    entry = next((e for e in man["versions"] if e["version"] == v), None)
    if entry is None:
        raise SystemExit(f"version {v} not found for style {name}")
    luts = {}
    for region, fname in (entry.get("regions") or {}).items():
        p = os.path.join(ws, name, fname)
        if os.path.exists(p):
            luts[region] = lc.read_cube(p)
    return luts, entry


def load_lut(ws: str, name: str, version: int | None = None) -> tuple[np.ndarray, dict]:
    man = load_manifest(ws, name)
    v = version or man.get("current")
    entry = next((e for e in man["versions"] if e["version"] == v), None)
    if entry is None:
        raise SystemExit(f"version {v} not found for style {name}")
    path = os.path.join(ws, name, entry["cube"])
    return lc.read_cube(path), {"manifest": man, "entry": entry, "path": path}


def blended_lut(lut: np.ndarray, strength: float) -> np.ndarray:
    if strength >= 0.999:
        return lut
    ident = lc.identity_lut(lut.shape[0])
    return np.clip(ident + (lut - ident) * float(strength), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- batches
# Every fit call is one batch. Batch samples are archived so that (a) a mis-merged
# batch can be undone without the original files, and (b) a periodic re-anchor can
# refit from everything instead of stacking LUT compositions.


def _batch_dir(ws: str, name: str) -> str:
    d = os.path.join(_dir(ws, name), "batches")
    os.makedirs(d, exist_ok=True)
    return d


def save_batch(ws: str, name: str, batch_id: str, samples: dict, meta: dict) -> dict:
    """samples: {'S': (N,3), 'T': (N,3), '<region>': (S,T)} — stored as float16."""
    d = _batch_dir(ws, name)
    payload = {}
    for key, val in samples.items():
        if isinstance(val, tuple):
            payload[f"{key}__S"] = np.asarray(val[0], dtype=np.float16)
            payload[f"{key}__T"] = np.asarray(val[1], dtype=np.float16)
        else:
            payload[f"{key}__S"] = np.asarray(val, dtype=np.float16)
    fname = f"{batch_id}.npz"
    np.savez_compressed(os.path.join(d, fname), **payload)
    entry = {"batch_id": batch_id, "file": f"batches/{fname}", **meta}
    man = load_manifest(ws, name)
    man.setdefault("batches", []).append(entry)
    with open(_manifest_path(ws, name), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return entry


def load_batches(ws: str, name: str) -> list:
    man = load_manifest(ws, name)
    return man.get("batches", [])


def merge_batch_samples(ws: str, name: str, region_names=()) -> dict:
    """Concatenate archived samples from every batch into {'S','T',region:(S,T)}."""
    out: dict = {"S": [], "T": []}
    for r in region_names:
        out[r] = [[], []]
    for b in load_batches(ws, name):
        p = os.path.join(ws, name, b["file"])
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            if "base__S" in z:
                out["S"].append(z["base__S"].astype(np.float32))
                out["T"].append(z["base__T"].astype(np.float32))
            for r in region_names:
                k = f"{r}__S"
                if k in z:
                    out[r][0].append(z[k].astype(np.float32))
                    out[r][1].append(z[f"{r}__T"].astype(np.float32))
    merged = {}
    if out["S"]:
        merged["S"] = np.concatenate(out["S"], axis=0)
        merged["T"] = np.concatenate(out["T"], axis=0)
    for r in region_names:
        if out.get(r) and out[r][0]:
            merged[r] = (np.concatenate(out[r][0], axis=0), np.concatenate(out[r][1], axis=0))
    return merged


def remove_batch(ws: str, name: str, batch_id: str) -> list:
    """Drop a batch from the manifest (file kept on disk so the undo itself is undoable)."""
    man = load_manifest(ws, name)
    before = man.get("batches", [])
    after = [b for b in before if b.get("batch_id") != batch_id]
    if len(after) == len(before):
        raise SystemExit(f"batch {batch_id} not found in style {name}")
    man["batches"] = after
    with open(_manifest_path(ws, name), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return after


# ---------------------------------------------------------------- scene tags


def update_tags(ws: str, name: str, scores: dict, weight: float = 1.0,
                floor: float = 0.25, top: int = 2) -> dict:
    """Running average of scene scores across every batch that built this style.

    A running average (not "last batch wins") matters because one portrait session
    should not relabel a style that is 90% landscape.
    """
    man = load_manifest(ws, name)
    prev = man.get("tag_scores") or {}
    prev_n = float(man.get("tag_weight") or 0.0)
    total = prev_n + max(weight, 1e-6)
    keys = set(prev) | set(scores)
    merged = {}
    for k in keys:
        merged[k] = (float(prev.get(k, 0.0)) * prev_n + float(scores.get(k, 0.0)) * weight) / total
    man["tag_scores"] = {k: round(v, 3) for k, v in
                         sorted(merged.items(), key=lambda kv: -kv[1])}
    man["tag_weight"] = round(total, 3)
    man["tags"] = [k for k, v in man["tag_scores"].items() if v >= floor][:top]
    with open(_manifest_path(ws, name), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return {"tags": man["tags"], "tag_scores": man["tag_scores"]}


def list_tags(ws: str) -> dict:
    out = {}
    for s in list_styles(ws):
        man = load_manifest(ws, s["style_id"])
        out[s["style_id"]] = {"tags": man.get("tags", []),
                              "tag_scores": man.get("tag_scores", {})}
    return out


def list_styles(ws: str) -> list:
    if not os.path.isdir(ws):
        return []
    out = []
    for n in sorted(os.listdir(ws)):
        p = os.path.join(ws, n, "style.json")
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            man = json.load(f)
        cur = man.get("current")
        entry = next((e for e in man["versions"] if e["version"] == cur), {})
        out.append({
            "style_id": n,
            "current_version": cur,
            "versions": len(man["versions"]),
            "size": entry.get("size"),
            "regions": sorted(entry.get("regions") or {}),
            "tags": man.get("tags", []),
            "tag_scores": man.get("tag_scores", {}),
            "batches": len(man.get("batches", [])),
            "look": (entry.get("description") or {}).get("summary"),
            "params": entry.get("params"),
            "metrics": entry.get("metrics"),
        })
    return out
