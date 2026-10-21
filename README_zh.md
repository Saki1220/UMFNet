# UMFNet

[English](README.md)

用于光学影像与 SAR 影像语义分割的 PyTorch 实现，包含模型、损失函数，以及面向预处理后 NumPy 数据的推理入口。

## 当前内容与验证范围

原始文件只包含模型、损失函数和权重。当前包未包含训练脚本、数据加载器、原始预处理、类别名称、数据划分或指标评估代码，尚未提供论文及引用信息。合成数据检查只验证代码执行和权重兼容性，不能证明分割精度或复现论文结果。

## 安装

本地验证环境：Python 3.13.10、PyTorch 2.9.1+cu126、NumPy 2.3.5；已测试 CPU 推理。其他环境尚未验证。

先安装适合本机 CPU/CUDA 环境的 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
```

## 权重与模型参数

`umfnet_whu.pth` 是直接保存的 `state_dict`，参数形状对应：

| 配置 | 数值 |
| --- | --- |
| 光学通道 | 4 |
| SAR 通道 | 1 |
| 输出通道／类别编号 | 8／0–7 |
| 构造参数 `n_classes` | 7 |

模型实际输出 `n_classes + 1` 类，因此必须使用：

```python
model = UMFNet(n_classes=7, opt_dim=4, sar_dim=1)
```

代码默认参数为 `n_classes=8, opt_dim=3`，与该权重不匹配。模型源代码保留原样，通过显式参数解决此问题。类别含义及光学波段顺序尚未提供；损失函数默认忽略标签 `0`，但不能仅据此确定真实数据集的标签映射。权重校验值和分发状态见 [WEIGHTS.md](WEIGHTS.md)。

## 推理

将已配准、已完成训练时预处理的光学和 SAR 输入分别保存为浮点 `.npy` 文件。默认形状为 `C × H × W`，SAR 也必须保留单通道维度；`H × W × C` 数据可使用 `--layout hwc`。

```bash
python infer.py --checkpoint umfnet_whu.pth --opt optical.npy --sar sar.npy --output results/prediction.npy --details results/details.npz --device cpu
```

脚本从权重读取通道数和类别数并严格加载，输入右侧及下侧以边缘复制补齐到 32 的倍数，输出裁剪回原尺寸。该补齐策略尚未与原始评估流程对照验证。脚本按整图推理，不包含滑窗，大图可能需要较多内存。

脚本不做归一化、波段排序、配准或 SAR 变换。结构模块用光学前三通道计算加权灰度，并包含与亮度有关的先验，所以必须使用正确的波段顺序和训练时预处理。现有文件不足以恢复这些信息。

输出 `prediction.npy` 为 `H × W` 的 int64 类别编号，不附带类别名称。可选 `details.npz` 包含 `logits`、`AU`、`EU`、`EU_mc`、`D_JSD`，形状均为 `C × H × W`。这些值未经过误差概率校准；`EU` 是 8 通道特征，与分割类别数不是同一概念。评估模式仍会执行随机通道掩码，脚本默认 `--seed 0`，但未验证跨设备的精确一致性。

## Python 接口

```python
import torch
from model import UMFNet

model = UMFNet(n_classes=7, opt_dim=4, sar_dim=1)
state = torch.load("umfnet_whu.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

# 以下仅为接口演示的合成数据。
with torch.inference_mode():
    outputs = model(torch.rand(1, 4, 32, 32), torch.rand(1, 1, 32, 32))
logits, enhanced_sar, au_terms, eu, auxiliary = outputs
prediction = logits.argmax(dim=1)
```

直接调用模型时，输入高宽需为 32 的倍数，两路数据尺寸、设备和类型需一致。默认返回分割 logits、增强 SAR、十个 AU 相关项、EU 特征及三个辅助分割输出。传入 `return_components=True` 时，还会返回含 `EU_mc` 和 `D_JSD` 的字典；详细顺序见英文说明。

`Loss` 的调用方式和训练时逐轮更新 `set_epoch()` 的示例见 [英文说明](README.md#loss-interface)。示例中的等权重只是接口演示，不能视为原始训练配置。

## 检查

```bash
python smoke_test.py
python smoke_test.py --checkpoint umfnet_whu.pth
```

检查模型输出形状与有限值、权重严格加载、全忽略标签交叉熵及合成张量上的损失反向计算；不包含完整模型训练或真实数据评估。

## 许可证与引用

源码、文档及提供的 `umfnet_whu.pth` 权重统一采用 [MIT 许可证](LICENSE)。论文信息尚未提供，当前未附论文引用。
