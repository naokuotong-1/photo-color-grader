# 第三方组件与许可

本仓库的代码以 MIT 许可发布，但**运行分区功能所需的模型权重不是本仓库的产物**，
也不随仓库分发。使用者需自行获取，并遵守其原始许可。

## SegFormer-B0（ADE20K 语义分割）

| 项 | 内容 |
|---|---|
| 来源 | [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) |
| 架构 | SegFormer-B0，ADE20K 150 类，512×512 输入 |
| 参数 | 3.75 M |
| 权重体积 | fp32 ONNX ≈ 14.6 MB；int8 量化 ≈ 4.2 MB |
| 许可 | HF 卡片标注为 `other`，对应 [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) |

**需要特别注意**：该许可对代码与权重允许研究、评估用途，
但**商业使用需另行取得 NVIDIA 授权**。因此本仓库不把 `models/segformer-b0-ade/*.onnx`
纳入版本控制，而是由使用者自行下载（见 `models/README.md`）。

如果你要在商业项目里用分区功能，请先确认 License 条款；
只用全局 LUT（`--regions ""`，不装 onnxruntime）不涉及这个模型，也不受此限制。

## 论文

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021).
*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.*
NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

## Python 依赖

| 包 | 许可 |
|---|---|
| numpy | BSD-3-Clause |
| pillow | HPND / MIT-CMU |
| onnxruntime | MIT |
| tifffile | BSD-3-Clause |
| rawpy / LibRaw | MIT / LGPL-2.1（LibRaw 为双重许可） |
