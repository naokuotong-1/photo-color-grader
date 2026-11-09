# 风格目录与 style.json

## 目录结构

```
color-styles/
└── warm_film/
    ├── style.json            # 版本清单，唯一的真相来源
    ├── v1.cube               # 33^3 3D LUT（基础层）
    ├── v1_sky.cube           # 同一版本的分区层（只有该分区启用时才有）
    ├── v2.cube
    ├── v3.cube               # current 指向它
    └── batches/
        ├── b20261108_0033.npz   # 这一批训练用的像素对（float16）
        └── b20261108_0107.npz
```

`.cube` 是标准 Resolve 格式，可直接给 DaVinci Resolve / Capture One / Premiere / OBS 使用。
**锚定空间为 sRGB（gamma 2.2）**，交付时必须一并说明。

分区层文件是**完整 LUT**（已经把基础层叠进去了），不是残差；所以它可以单独替换基础层用，
但正确用法是「基础层铺全图 + 分区层在软蒙版里叠加」。

## style.json

```json
{
  "style_id": "warm_film",
  "lut_space": "srgb-gamma-2.2",
  "current": 3,
  "tags": ["street", "landscape"],
  "tag_scores": {"street": 0.99, "landscape": 0.61, "indoor": 0.0, "night": 0.0, "portrait": 0.0},
  "tag_weight": 2,
  "versions": [
    {
      "version": 3,
      "cube": "v3.cube",
      "size": 33,
      "created": "2026-11-08T00:33:13",
      "params": {
        "mode": "fit",
        "pairs": 1,
        "samples": 90000,
        "monotonic": true,
        "lut_space": "srgb-gamma-2.2",
        "regions": ["sky"],
        "region_strength": 0.9,
        "batches_merged": 2
      },
      "metrics": {
        "samples_used": 14903,
        "coverage_pct": 9.21,
        "prior": "affine+1d-curve",
        "prior_shrink": 1.0,
        "prior_trials": [[1.0, 0.61], [0.5, 0.93], [0.2, 1.24], [0.05, 1.42]],
        "lut_clipped_pct_all": 2.33,
        "lut_clipped_pct_observed": 0.26,
        "dE2000_mean": 0.554,
        "dE2000_p95_mean": 1.423,
        "regions": {
          "sky": {"pixels": 91116, "max_dev_from_base": 0.166, "coverage_pct": 1.46,
                  "clipped_pct_observed": 0.0}
        }
      },
      "key": {
        "src": {"p1": 0.002, "p10": 0.006, "p50": 0.195, "p90": 0.271, "p99": 0.353},
        "tgt": {"p1": 0.022, "p10": 0.035, "p50": 0.284, "p90": 0.464, "p99": 0.564}
      },
      "regions": {"sky": "v3_sky.cube"},
      "description": {
        "summary": "中间调对比减 2%；高光收 9.6 L*",
        "measured": {"exposure_ev": -0.007, "contrast_slope": 0.978,
                     "highlight_L_delta": -9.61, "shadow_L_delta": 0.01,
                     "saturation_ratio": 1.01, "hue_shift_deg": 0.08,
                     "grey_a_shift": 0.18, "grey_b_shift": -0.35},
        "estimate_lightroom": {"Exposure": -0.01, "Contrast": -5, "Highlights": -48,
                               "Shadows": 0, "Saturation": 1, "Temperature": -1, "Tint": 0},
        "caveat": "measured 一栏可直接复现；estimate_lightroom 是估算，LUT 到滑块没有唯一解",
        "note": "measured 描述的是全局基础层；实际渲染会在对应区域叠加分区层",
        "region_extra": {"sky": "对比再减 19%；高光再收 14.5 L*"}
      }
    }
  ]
}
```

### 字段语义

| 字段 | 说明 |
|---|---|
| `current` | 默认使用的版本号；`apply` 不带 `--version` 时取它 |
| `versions[].cube` | 相对风格目录的文件名，不要改成绝对路径 |
| `versions[].params.mode` | `fit` 或 `reference`，回看时能知道这版怎么来的 |
| `versions[].params.regions` | 这一版启用了哪些分区；空/缺省 = 纯全局风格 |
| `versions[].params.region_strength` | 分区残差保留比例（0..1），越小越接近全局风格 |
| `versions[].params.batches_merged` | 这一版是几批素材合并拟合出来的 |
| `versions[].metrics` | 生成时的诊断快照，用于比较「新版是不是真的更好」 |
| `versions[].regions` | `{区域名: cube 文件名}`，缺省表示没有分区层 |
| `versions[].key.src` | 训练素材（原图）的亮度分位。`apply --match-key` 用它判断这张图要不要先对曝光 |
| `versions[].key.tgt` | 成片的亮度分位，描述这个 look 自己的白点在哪（实测 p99 0.56 就是它没有高光余量） |
| `versions[].description` | 给用户看的风格描述；`measured` 与 `estimate_lightroom` 必须分开标注 |
| `tags` / `tag_scores` | 从训练图自动归纳的题材标签，`apply --style auto` 用它选风格 |
| `tag_weight` | 已累计的标签权重（批次数），用于滚动平均 |
| `lut_space` | 锚定空间。改动它等于换了颜色语义，**不要手改** |

### 为什么描述层要分两栏

`measured` 是从 LUT 直接量出来的（曝光 EV、对比斜率、高光/阴影 L* 位移、饱和度比、
色相旋转、中性灰 a\*/b\* 偏移），可复现；`estimate_lightroom` 是把这些量近似折算成
Lightroom 滑块值，**一个 LUT 通常对应多组滑块组合**，所以只能当起点参考。
两栏混在一起会让用户以为滑块值能原样复刻，必须显式标注为估算。

### `key` 是用来判断「这个 look 能不能套」的

`key` 不是元数据装饰。一个 look 的亮度分位决定了它在什么明暗范围内可信：
套到比 `key.src` 亮得多的图上，超出 p99 的部分只能靠线性外推，高光会顶穿。
`apply` 会把 `look.median_luma` 的前后值打出来，用户看到 `[0.55, 0.95]` 就该停下来看 `key`。
`--match-key on` 会先用中位亮度把画面拉回 `key.src.p50` 再套，只适合同光比的连拍——
逆光片按全画面中位对齐会把主体压死，详见 SKILL.md 保真红线第 14 条。

## 版本管理约定

- 版本号只增不减，永不覆盖。回滚靠 `--version` 指定，不靠删除文件。
- 每次重新 `fit` 都会追加新版本并自动把 `current` 指过去。
- `undo` 不是删除版本：它删掉**一批训练素材**，再从剩下的素材重新拟合，产出**新版本**。
  因此撤销后的 vN+1 与撤销前的 vN-1 应当几乎一致（实测差异 <0.002 ΔE）。
- 用户否定新版时，不要删 vN，改用 `--version` 回退，并把这一版的指标记下来做对比。
- 风格名建议 `用途_色系_序号`，例如 `portrait_warm_02`，避免 `test`、`new` 这种不可检索的名字。

## 批次归档

`batches/*.npz` 保存这一批训练实际用到的像素对（float16，键名 `<name>__S` / `<name>__T`，
基础层用 `base__S` / `base__T`）。作用有三个：

1. `undo` 需要从剩余素材重新拟合，没有归档就无法撤销。
2. 合并新素材时是全量重拟合，不是把两个 LUT 叠在一起——后者会累积插值误差。
3. 归档体积远小于原图，长期累积不会失控。

## 迁移与分享

- 整个风格目录可直接打包，体积约 1–9 MB（33³ 约 1.3 MB，65³ 约 11 MB，另加 batches）
- 单独分享某个版本时只给对方 `vN.cube` 会**丢掉分区和描述**。分区风格请用
  `grade.py export --full` 一次导出基础层 + 分区层 + 接线说明。
- 分区风格无法合并成单个 `.cube`：单个 3D LUT 是像素颜色的函数，
  表达不了「同一颜色在不同区域不同处理」。
