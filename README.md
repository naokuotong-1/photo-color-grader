# photo-color-grader

**中文** | [English](README.en.md)

把「一种调色风格」学成一个可复用的 3D LUT，然后批量套到新素材上。
逐像素色彩映射，不重绘、不改变内容、不改几何——结构 SSIM 恒等于 1.0。

A skill that learns a color grade as a reusable 3D LUT and applies it in batch.
Pure per-pixel color mapping: no inpainting, no content change, SSIM ≡ 1.0.
Region-aware (`sky` / `people` / `veg` / …) via SegFormer-B0 soft masks.

---

## 它解决什么问题

生成式修图擅长改内容，但用在「交付级调色」上有两个硬伤：会动主体、会掉细节。
而一个 3D LUT 只回答一个问题——**这个颜色映射成什么**。所以：

- 输出尺寸和输入严格一致，几何不变
- 同一张图跑两次，结果逐像素相同
- 风格以 `.cube` 形式存在，能直接导入达芬奇 / Lightroom / Premiere

### 为什么需要分区

全局 LUT 遇到「同一颜色在不同区域要被处理成不同结果」时，数学上无解。
最典型的情况是**灰白的天空要压暗，而同样灰白的路面要保留**——最小二乘只会取中间值，两边都不像。

实测（一张阴天街景，只压暗天空 −28%）：

| | 天空 ΔE2000 | 非天空 ΔE2000 |
|---|---|---|
| 全局 LUT | **11.27** | 0.322 |
| 全局 + sky 分区 | **1.50** | 0.324 |

非天空误差完全一致，说明分区没有引入额外污染，只是把「学不会的那部分」补上了。

## 核心特性

- **两种风格来源**：`fit` 从「原图 / 修后图」配对学习，精确（实测 ΔE2000 0.55–0.9，需 ≥3 对）；`ref` 从参考图统计迁移，粗略（≈4）但单张图即可，适合冷启动
- **提交即沉淀**：每次 `fit` 自动判定属于哪个风格 → 并入则用全部历史样本重拟合 → 存档 → 打题材标签。不训练模型，不需要 GPU
- **分区调整**：`sky` / `people` / `veg` / `water` / `ground` / `arch`，由内置 SegFormer-B0 给软掩膜；缺失时自动降级为全局，不报错
- **无参考自检**：`check` 不需要参考图，回答「这是什么素材」和「成片有没有硬伤」，能识别 log 未转色素材
- **可撤销**：`undo` 把某一批次从构成里剔除并重拟合，旧版本永远保留
- **交付级输出**：ICC / EXIF / DPI 从源图透传，JPEG q98 + 无损母版双落盘，过曝区软保护

## 安装

这是一个 [Agent Skill](https://www.workbuddy.cn/docs/workbuddy/Overview)（`SKILL.md` + `scripts/`）。

```bash
git clone https://github.com/naokuotong-1/photo-color-grader.git

# WorkBuddy
cp -r photo-color-grader ~/.workbuddy/skills/
# 或 Claude Code
cp -r photo-color-grader ~/.claude/skills/
```

依赖：

```bash
python -m pip install numpy pillow onnxruntime   # 分区功能需要 onnxruntime
python -m pip install tifffile rawpy             # 可选：16bit TIFF / RAW 直读
```

分区权重不随仓库分发（许可与体积原因），单独取：

```bash
python scripts/fetch_model.py                    # 约 4.2 MB
```

不装也能跑，只是退回纯全局调色。详见 [`models/README.md`](models/README.md)。

## 快速开始

```bash
S=./scripts/grade.py

# 1. 先看一眼素材：是 log 还是显示参考？成片有没有硬伤？（不需要参考图）
python $S check --in ./photos/original --person

# 2. 从「原图 / 修后图」配对学到风格（按文件名主干配对）
python $S fit --src ./photos/original --tgt ./photos/retouched

# 3. 套到新素材上
python $S apply --style auto --in ./photos/new --out ./out --strength 1.0

# 4. 校验
python $S verify --style warm_film \
  --src ./photos/original/a.jpg --tgt ./photos/retouched/a.jpg --threshold 2.0

# 5. 导出给达芬奇 / Lightroom
python $S export --style warm_film --out ./for_lightroom --full
```

所有命令往 stdout 打印 JSON，便于链式和二次处理。

## 命令参考

| 命令 | 作用 |
|---|---|
| `fit` | 从配对图学习风格 LUT，自动新建或并入已有风格，自动存档 |
| `ref` | 从单张参考图迁移风格，快速但粗略（ΔE ≈ 4） |
| `apply` | 批量渲染，支持 `--strength`、`--version`、`--style auto` |
| `verify` | LUT 自检，或对着修后图算 dE2000；带 `--threshold` 时超阈值退出码 2 |
| `check` | 无参考自检：素材类型（display / ambiguous / log）+ 成片硬伤 |
| `styles` | 列出全部风格、标签、一句话描述 |
| `batches` | 查看某个风格由哪些批次构成 |
| `undo` | 剔除某批次并用剩余批次重拟合，生成新版本 |
| `export` | 写出 `.cube`；`--full` 同时给分区层和接线说明 |

常用参数：

```
fit     --regions sky,people,veg   --auto-threshold 1.5   --size 33   --no-monotonic
apply   --strength 0.6             --match-key off        --master png|tif|none   --quality 98
fit/apply/check                          --seg-full      # 用 fp32 权重而非 int8
```

### 输出产物

默认每次落两个文件：

| 产物 | 默认 | 用途 |
|---|---|---|
| `<名>__<风格>_v<N>.jpg` | q98、4:4:4、保留源 ICC/EXIF/DPI | 交付、发人 |
| `<名>__<风格>_v<N>__master.png` | 8bit 无损、带 ±0.5 LSB 抖动 | 继续修、放大、印刷 |

`--master tif` 走 16bit（需要 `tifffile`，装不上会自动降级成 8bit PNG 并写 `master_note`）。
批量出图嫌占地方用 `--master none`。

## 读输出时要看的字段

| 字段 | 判断 |
|---|---|
| `quality.dE2000_mean` | < 2 可用；2–4 勉强；> 4 说明配对不是纯色彩关系 |
| `decision` | `create-new` / `merge-nearest` / `merge-into-named` |
| `diagnostics.lut_clipped_pct_observed` | > 2% 要提示高光/暗部可能丢失 |
| `regions.active` | 为空说明这些区域在本批素材里没出现 |
| `health.monotonic_violations` | 默认强制归零，非零说明 LUT 会出色彩断层 |
| `fit_quality.geometry_preserved` | 必须为 true |
| `transfer.verdict` | `log` 必须先转色再 fit；`ambiguous` 让用户确认来源 |
| `look` | `[原图, 成片]` 的中位亮度 / 死白比例 / 饱和度 |
| `warnings[]` | 无参考自检抓到的问题，空数组就是没发现硬伤 |

## 已知局限

完整 17 条见 [`SKILL.md`](SKILL.md) 的「保真红线」。最需要先知道的：

1. **风格带曝光属性**。LUT 只在它学过的明暗范围里可信。把中位 0.19 学到 0.28 的 look，套到中位 0.55 的片上会冲到 0.95——不是 bug，是 look 没有那么多高光余量。这时收 `--strength`，别怪算法
2. **`ref` 不是「精准复制」**。它只对齐 Lab 分布，实测 ΔE ≈ 4。要精准必须走 `fit`
3. **一个 `.cube` 表达不了分区**。分区层在 `vN_<region>.cube`，要配软蒙版；导出用 `export --full`
4. **掩膜只有 128×128**。细于 1/128 画幅的枯枝、电线会被算进所在分区。要像素级精确请用外部蒙版
5. **过曝像素原样通过**。渲染末尾对最亮通道 ≥0.97 的像素软过渡回原图：过曝区没有颜色信息，任何映射都是发明
6. **RAW 是近似解码**。libraw 用相机白平衡 + 通用色彩矩阵，不用相机 DCP。要精确交付请导 16bit TIFF
7. **LUT 锚定 sRGB gamma 2.2**。导出给别人时要说明，否则对方套错空间颜色全错
8. **无参考自检只能发现「明显的错」**，判断不了「好不好看」。告警的含义是「看一眼」，不是「错了」

## 仓库结构

```
SKILL.md                     技能说明与全部 17 条保真红线（Agent 读这个）
references/
  color-pipeline.md          色彩管理、算法细节、调参建议
  style-schema.md            风格目录、style.json 字段、版本与批次语义
scripts/
  grade.py                   CLI 入口（fit / ref / apply / verify / check / export / undo / batches / styles）
  lut_core.py                LUT 拟合、采样、分区合成、渲染核心
  seg.py                     SegFormer ONNX 推理与软掩膜
  imgcheck.py                无参考自检（素材类型 + 成片统计）
  style_store.py             带版本的风格库与批次档案
  fetch_model.py             下载分区权重
models/                      权重目录（不随仓库分发，见其 README）
```

## 性能

10.8 MP 图，单核 CPU：

| 场景 | 耗时 |
|---|---|
| 纯全局渲染 | 3.6 s |
| 分区渲染（2 个区域生效） | 6.7 s |
| 分割推理（512px）+ 上采样 | 0.7 s |
| 一次 `fit`（10 对图，含分区与打标） | ≈ 12 s |

瓶颈在逐像素 LUT gather（8 次随机读 × 每像素），不在算法分支。批量处理按图多进程并行最划算。

## 许可

代码与文档 MIT，见 [LICENSE](LICENSE)。
分区权重来自 NVIDIA SegFormer，许可是「非商业研究用途」，不随仓库分发，
商业使用前请确认条款：见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
只用全局 LUT（`--regions ""`）不涉及该模型。
