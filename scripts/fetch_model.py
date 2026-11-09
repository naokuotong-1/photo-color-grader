#!/usr/bin/env python
"""下载分区功能所需的 SegFormer-B0 (ADE20K) ONNX 权重。

默认取 int8 量化版权重（约 4.2 MB），加 --full 取 fp32 版（约 14.6 MB）。
国内网络慢可以先用镜像：

    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_model.py

脚本只做下载与落盘，不做任何转换。下载后可用的文件布局：

    models/segformer-b0-ade/
        config.json
        preprocessor_config.json
        model_quantized.onnx   # 默认
        model.onnx             # --full 时才有
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

REPO = "nvidia/segformer-b0-finetuned-ade-512-512"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "segformer-b0-ade")
CONFIGS = ["config.json", "preprocessor_config.json"]


def _endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "photo-color-grader/fetch_model"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, dst: str, label: str) -> None:
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "photo-color-grader/fetch_model"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stderr.write(f"\r  {label}: {pct:3d}%  {done / 1e6:5.1f}/{total / 1e6:.1f} MB")
            else:
                sys.stderr.write(f"\r  {label}: {done / 1e6:5.1f} MB")
        sys.stderr.write("\n")
    os.replace(tmp, dst)


def _list_files(endpoint: str) -> list[str]:
    data = json.loads(_get(f"{endpoint}/api/models/{REPO}"))
    return [s["rfilename"] for s in data.get("siblings", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description="download SegFormer-B0 ADE20K ONNX weights")
    ap.add_argument("--full", action="store_true", help="prefer fp32 model.onnx over int8")
    ap.add_argument("--dir", default=TARGET_DIR, help=f"target dir (default: {TARGET_DIR})")
    args = ap.parse_args()

    endpoint = _endpoint()
    os.makedirs(args.dir, exist_ok=True)
    want = "model.onnx" if args.full else "model_quantized.onnx"

    try:
        files = _list_files(endpoint)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        print(f"无法访问 {endpoint}（{exc}）", file=sys.stderr)
        print("手动下载：见 models/README.md", file=sys.stderr)
        return 2

    weights = {os.path.basename(p): p for p in files if p.endswith(".onnx")}
    if want not in weights:
        print(f"仓库里没有 {want}，只有 {sorted(weights)}", file=sys.stderr)
        print("手动下载：见 models/README.md", file=sys.stderr)
        return 2

    plan = [(weights[want], os.path.join(args.dir, want))]
    for extra in ("model_quantized.onnx", "model.onnx"):
        if extra != want and extra in weights:
            plan.append((weights[extra], os.path.join(args.dir, extra)))
    plan += [(c, os.path.join(args.dir, c)) for c in CONFIGS if c in files]

    for path, dst in plan:
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"跳过（已存在）{dst}")
            continue
        print(f"下载 {path} -> {dst}")
        try:
            _download(f"{endpoint}/{REPO}/resolve/main/{path}", dst, os.path.basename(dst))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            if os.path.exists(dst + ".part"):
                os.remove(dst + ".part")
            print(f"失败：{exc}", file=sys.stderr)
            return 3

    ok = os.path.exists(os.path.join(args.dir, want)) and all(
        os.path.exists(os.path.join(args.dir, c)) for c in CONFIGS
    )
    if not ok:
        print("文件不齐，检查上面的输出。手动步骤见 models/README.md", file=sys.stderr)
        return 4

    total = sum(os.path.getsize(os.path.join(args.dir, f)) for f in os.listdir(args.dir))
    print(f"完成：{args.dir}（{total / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
