# photo-color-grader

[中文](README.md) | **English**

Learn a color grade as a reusable 3D LUT, then apply it to a new batch of frames.
Pure per-pixel color mapping — no inpainting, no content change, no geometry change:
structural SSIM is identically 1.0.
Region-aware (`sky` / `people` / `veg` / `water` / `ground` / `arch`) via SegFormer-B0 soft masks.

Shipped as an [Agent Skill](https://www.workbuddy.cn/docs/workbuddy/Overview) — a `SKILL.md` plus a CLI.
It runs inside WorkBuddy or Claude Code, and `scripts/grade.py` also works as a plain command-line tool
with no agent involved.

> **Note for English readers** — `SKILL.md`, the files under `references/`, and the human-readable
> strings in the JSON output (`description.summary`, `warnings[].note`) are written in Chinese.
> The CLI flags, field names, and numeric output are language-neutral. If you need English
> docs or English output strings, open an issue.

---

## What it is for

Generative retouching is good at changing content and bad at delivery-grade grading: it nudges the
subject and throws away fine detail. A 3D LUT only ever answers one question — **what does this color
map to** — which buys three properties that matter for delivery:

- output dimensions are strictly identical to the input; geometry is untouched
- the same frame graded twice yields pixel-identical results
- the grade lives as a `.cube`, importable into DaVinci Resolve / Lightroom / Premiere

### Why regions are needed

A global LUT cannot express "the same color should land in a different place depending on where it
is". The canonical case: a flat grey-white **sky** you want to darken, and a road of exactly the same
grey-white you want left alone. Least squares splits the difference and pleases neither.

Measured on one overcast street scene, darkening the sky by 28% and nothing else:

| | sky ΔE2000 | non-sky ΔE2000 |
|---|---|---|
| global LUT | **11.27** | 0.322 |
| global + sky region | **1.50** | 0.324 |

The non-sky error is unchanged to three digits — the region warp adds no contamination, it only
supplies the part a global map mathematically cannot learn.

## What you get

- **Two ways to build a look.** `fit` learns from (original, retouched) pairs and is accurate
  (measured ΔE2000 0.55–0.9, needs 3+ pairs). `ref` transfers color statistics from a single
  reference image — rough (ΔE ≈ 4) but enough to cold-start.
- **Every `fit` is a commit.** The tool decides which style it belongs to, refits on all historical
  samples when merging, archives the batch, and tags the subject matter. No model training, no GPU.
- **Region-aware grading** over six semantic regions with soft masks. Falls back to global grading
  silently when the weights are absent — it never crashes.
- **Reference-free QC.** `check` needs no reference image: it tells you what kind of frame this is
  and whether the result has obvious defects, including detecting log footage that was never converted.
- **Undo is real undo.** `undo` drops one batch from the style's composition and refits from the
  rest. Old versions are never destroyed.
- **Delivery-grade output.** ICC / EXIF / DPI carried through from the source, JPEG q98 plus a
  lossless master, and a soft guard that passes blown highlights through unchanged.

## Install

```bash
git clone https://github.com/naokuotong-1/photo-color-grader.git

cp -r photo-color-grader ~/.workbuddy/skills/   # WorkBuddy
cp -r photo-color-grader ~/.claude/skills/      # or Claude Code
```

Dependencies:

```bash
python -m pip install numpy pillow onnxruntime   # onnxruntime enables region grading
python -m pip install tifffile rawpy             # optional: 16-bit TIFF / RAW input
```

The segmentation weights are **not** in this repo (size, and the upstream license is
non-commercial research use). Fetch them separately:

```bash
python scripts/fetch_model.py                    # ~4.2 MB, int8
HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_model.py   # behind the GFW
```

Skip this step and everything still runs, just without regions. See [`models/README.md`](models/README.md).

## Quick start

```bash
S=./scripts/grade.py

# 1. What kind of frame is this? Is the result broken? (no reference needed)
python $S check --in ./photos/original --person

# 2. Learn a style from (original, retouched) pairs, matched by filename stem
python $S fit --src ./photos/original --tgt ./photos/retouched

# 3. Apply it to new material
python $S apply --style auto --in ./photos/new --out ./out --strength 1.0

# 4. Check the learned look against a held-out pair
python $S verify --style warm_film \
  --src ./photos/original/a.jpg --tgt ./photos/retouched/a.jpg --threshold 2.0

# 5. Export for Resolve / Lightroom
python $S export --style warm_film --out ./for_lightroom --full
```

Every command prints a JSON object on stdout, so they chain cleanly.

## Commands

| Command | What it does |
|---|---|
| `fit` | Learn a LUT from paired images; creates a style or merges into an existing one, auto-archived |
| `ref` | Transfer a look from a single reference image — fast but rough (ΔE ≈ 4) |
| `apply` | Batch render; supports `--strength`, `--version`, `--style auto` |
| `verify` | LUT self-check, or dE2000 against a retouched image; exits 2 past `--threshold` |
| `check` | Reference-free inspection: footage type (display / ambiguous / log) plus defect statistics |
| `styles` | List every style with tags and a one-line description |
| `batches` | Show which batches compose a style |
| `undo` | Drop one batch, refit from the rest, emit a new version |
| `export` | Write `.cube` files; `--full` also emits the region layers and wiring notes |

Flags worth knowing:

```
fit     --regions sky,people,veg   --auto-threshold 1.5   --size 33   --no-monotonic
apply   --strength 0.6             --match-key off        --master png|tif|none   --quality 98
fit / apply / check                       --seg-full      # fp32 weights instead of int8
```

### Output files

Two files per frame by default:

| File | Default | Use |
|---|---|---|
| `<name>__<style>_v<N>.jpg` | q98, 4:4:4, source ICC/EXIF/DPI preserved | delivery, sharing |
| `<name>__<style>_v<N>__master.png` | 8-bit lossless, ±0.5 LSB dither | further editing, enlarging, print |

`--master tif` writes 16-bit (needs `tifffile`; falls back to 8-bit PNG with a note in `master_note`
rather than failing). `--master none` if you're producing enough frames to care about disk.

## Reading the output

| Field | How to judge it |
|---|---|
| `quality.dE2000_mean` | < 2 usable; 2–4 marginal; > 4 means the pair isn't a pure color relationship |
| `decision` | `create-new` / `merge-nearest` / `merge-into-named` |
| `diagnostics.lut_clipped_pct_observed` | > 2% means highlights or shadows may be lost |
| `regions.active` | Empty means none of those regions appeared in this batch |
| `health.monotonic_violations` | Forced to zero by default; non-zero means visible banding |
| `fit_quality.geometry_preserved` | Must be `true` |
| `transfer.verdict` | `log` must be converted before `fit`; `ambiguous` needs a human call on the source |
| `look` | `[before, after]` median luma / clipped-white ratio / saturation |
| `warnings[]` | Defects found by reference-free QC — empty means nothing obvious |

## Known limitations

The full 17-item list lives in [`SKILL.md`](SKILL.md) (Chinese). The ones you'll hit first:

1. **A style carries its exposure range.** The LUT is only trustworthy inside the brightness range it
   was fitted on. A look that lifts median 0.19 → 0.28 applied to a frame with median 0.55 will push it
   to 0.95. Not a bug — the look has no highlight headroom to spend. Lower `--strength`.
2. **`ref` is not "exact copy".** It aligns Lab distributions only, measured ΔE ≈ 4. Use `fit` when
   accuracy matters.
3. **One `.cube` cannot express regions.** Region layers live in `vN_<region>.cube` and need a soft
   mask to drive them. Use `export --full`.
4. **Masks are 128×128.** Structures finer than about 1/128 of the frame — twigs, wires, thin railings —
   get folded into whichever region they sit in. Use an external matte if you need pixel accuracy.
5. **Blown pixels pass through.** At the end of rendering, pixels whose brightest channel is ≥ 0.97 are
   softened back toward the original: a blown area holds no color information, so any mapping is
   invention.
6. **RAW is an approximate decode.** libraw uses camera white balance plus a generic color matrix, not
   the camera's DCP. Export 16-bit TIFF when accuracy matters.
7. **The LUT is anchored in sRGB gamma 2.2.** Say so when handing a `.cube` to someone, or they'll
   apply it in the wrong space and every color will be wrong.
8. **Reference-free QC finds obvious errors, not bad taste.** It uses classical statistical targets
   (median 0.45, black point 0.02, skin hue 20°) with tolerance bands, because night shots *should* be
   dark and sunsets *should* be warm. A warning means "take a look", not "this is wrong".

## Repository layout

```
SKILL.md                     skill definition + all 17 fidelity rules (Chinese; the agent reads this)
references/
  color-pipeline.md          color management, algorithms, tuning notes
  style-schema.md            style directory, style.json fields, version and batch semantics
scripts/
  grade.py                   CLI entry point
  lut_core.py                fitting, sampling, region compositing, rendering
  seg.py                     SegFormer ONNX inference and soft masks
  imgcheck.py                reference-free QC
  style_store.py             versioned style store and batch archive
  fetch_model.py             weight downloader
models/                      weights directory (not distributed; see its README)
```

## Performance

10.8 MP frame, single-threaded CPU:

| Case | Time |
|---|---|
| Global render only | 3.6 s |
| Region render (2 active regions) | 6.7 s |
| Segmentation (512 px) + upsample | 0.7 s |
| One `fit` over 10 pairs, incl. regions and tagging | ≈ 12 s |

The bottleneck is the per-pixel LUT gather (8 random reads per pixel), not control flow.
For batches, parallelise across images with a process pool.

## License

Code and documentation are MIT — see [LICENSE](LICENSE).

The segmentation weights come from NVIDIA SegFormer and are licensed for non-commercial research
use; they are not distributed here. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
commercial use. Global-only grading (`--regions ""`, no onnxruntime) does not involve that model.
