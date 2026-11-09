# 模型权重（不随仓库分发）

分区调色（`sky` / `people` / `veg` / `water` / `ground` / `arch`）和题材标签依赖一个
SegFormer-B0 的 ONNX 推理。权重约 4–20 MB，且上游许可为「非商业研究用途」，
所以不放进取版本库。

**不装也能用**：缺 onnxruntime 或缺权重时，脚本会自动降级成纯全局 LUT，
只输出一条提醒，不会报错中断。纯全局模式对「同一颜色映射成同一结果」的场景完全够用。

## 一条命令

```bash
python scripts/fetch_model.py            # int8 量化版，约 4.2 MB（推荐）
python scripts/fetch_model.py --full     # fp32 版，约 14.6 MB，略准、慢 2~3 倍
```

国内网络：

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_model.py
```

脚本从 Hugging Face 的模型仓库按文件名探测并下载，落盘到
`models/segformer-b0-ade/`，已存在的文件会跳过。

## 手动下载

从 <https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/main>
下载以下文件，放到 `models/segformer-b0-ade/`，文件名保持一致：

```
config.json                  # 必需，150 类 id2label
preprocessor_config.json     # 必需
model_quantized.onnx         # int8 版，体积小、默认使用
model.onnx                   # fp32 版，可选，--seg-full 时才读
```

ONNX 文件通常在仓库的 `onnx/` 子目录下，`config.json` 在根目录。

## 校验

```bash
python -c "import onnxruntime as ort, os; p='models/segformer-b0-ade/model_quantized.onnx'; s=ort.InferenceSession(p, providers=['CPUExecutionProvider']); print(s.get_inputs()[0].shape)"
```

打印出 `[1, 3, 512, 512]` 或带动态维度的四维形状即可。

## 许可

权重来自 `nvidia/segformer-b0-finetuned-ade-512-512`，许可见
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。商业使用前请自行确认条款。
