# UMFNet

[中文说明](README_zh.md)

PyTorch implementation of UMFNet for optical/SAR semantic segmentation, including model components, the training loss, and an inference utility for prepared NumPy inputs.

## Release scope

This package contains the supplied model and loss implementation. It does not include a training pipeline, dataset loader, original preprocessing, label names, dataset splits, or benchmark evaluation code. Paper metadata and citation information have not yet been supplied. Synthetic checks establish execution and checkpoint compatibility; they do not establish segmentation accuracy or reproduce a paper's results.

## Installation

Local verification used Python 3.13.10, PyTorch 2.9.1+cu126, and NumPy 2.3.5. CPU inference was tested. Other environments have not been verified.

Install the PyTorch build appropriate for your CPU/CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Checkpoint

The supplied `umfnet_whu.pth` is a plain state dictionary. Its shapes require:

| Setting | Value |
| --- | --- |
| Optical channels | 4 |
| SAR channels | 1 |
| Output channels / class IDs | 8 / 0–7 |
| Constructor `n_classes` | 7 |

The model creates `n_classes + 1` output channels. Its constructor defaults (`n_classes=8`, `opt_dim=3`) therefore do not match this checkpoint. Preserve the implementation and use the explicit parameters below. Class meanings and the optical band order are not available in the supplied files. The loss defaults to ignoring label `0`; this does not by itself establish the dataset's label mapping.

See [WEIGHTS.md](WEIGHTS.md) for the checkpoint hash and distribution status. Weights are kept outside Git history.

## Python usage

```python
import torch
from model import UMFNet

model = UMFNet(n_classes=7, opt_dim=4, sar_dim=1)
state = torch.load("umfnet_whu.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

# Synthetic inputs demonstrate the interface only.
optical = torch.rand(1, 4, 32, 32)
sar = torch.rand(1, 1, 32, 32)
with torch.inference_mode():
    logits, enhanced_sar, au_terms, eu, auxiliary = model(optical, sar)
    prediction = logits.argmax(dim=1)
```

Direct model calls require spatial dimensions divisible by 32. Both inputs must have matching dimensions, device, and dtype, with the preprocessing used during training already applied. The structural module uses the first three optical channels with grayscale coefficients `(0.299, 0.587, 0.114)` and contains intensity-dependent priors. Band order, scaling, and SAR transformations therefore matter; they cannot be reconstructed from weights alone.

## Command-line inference

Save each co-registered input as one floating-point `.npy` array. Default layout is `C × H × W`; use `--layout hwc` for `H × W × C`. Supply all four optical bands and one SAR band for this checkpoint, including an explicit channel dimension for SAR. The script does not normalize, reorder bands, register images, or infer the training preprocessing.

```bash
python infer.py --checkpoint umfnet_whu.pth --opt optical.npy --sar sar.npy --output results/prediction.npy --details results/details.npz --device cpu
```

The utility reads architecture dimensions from the checkpoint and loads it strictly. It pads the bottom/right borders to multiples of 32 using replication, then crops outputs to the original size. Padding is an inference convenience and has not been validated against the original evaluation protocol. Whole-image inference may need substantial memory; tiled inference is not included.

`prediction.npy` contains an `H × W` int64 mask of raw class indices. Optional `details.npz` contains `logits`, `AU`, `EU`, `EU_mc`, and `D_JSD` in `C × H × W` layout. These are model outputs, not calibrated probabilities of error. `EU` has eight feature channels, independently of the number of segmentation classes. MC channel masking remains stochastic in evaluation mode; `--seed` defaults to `0`, but exact reproducibility across hardware is not established.

## Forward interface

`model(optical, sar)` returns:

1. Segmentation logits, `[B, n_classes + 1, H, W]`.
2. Enhanced SAR, `[B, 1, H, W]`.
3. Ten AU-related terms: `mu`, `AU`, `log_sar`, `log_opt`, `rho_opt`, `rho_sar`, `eta_opt`, `eta_sar`, `eta_null`, `align_conf`.
4. Combined EU features, `[B, 8, H, W]`.
5. Three auxiliary segmentation logits: optical, SAR, and fused.

With `return_components=True`, a sixth item contains `EU_mc` and `D_JSD`.

## Loss interface

```python
import numpy as np
from losses import Loss

# Illustrative equal weights; replace with your training configuration.
criterion = Loss(np.ones(8, dtype=np.float32), device="cpu", ignore_label=0)
criterion.set_epoch(1)  # Call once per epoch for uncertainty warmup.
# labels: int64 [B, H, W], with values in 0..7 for this checkpoint.
# outputs must be computed with gradients enabled during training.
# logits, enhanced_sar, au_terms, eu, auxiliary = model(optical, sar)
# loss = criterion(labels, logits, [sar, enhanced_sar], au_terms, auxiliary)
# loss.backward()
```

This demonstrates the loss signature only. It is not a training recipe or an assertion about the original class weights, optimizer, or schedule.

## Verification

```bash
python smoke_test.py
python smoke_test.py --checkpoint umfnet_whu.pth
```

These CPU checks cover model output shapes and finite values, strict checkpoint compatibility when supplied, all-ignored-label cross entropy, and loss backward on synthetic tensors. They do not run full model training or evaluate real data.

## Files

```text
model/          Original model, backbone, and fusion components
losses.py       Original segmentation and uncertainty losses
infer.py        NumPy inference utility
smoke_test.py   Synthetic execution checks
requirements.txt
WEIGHTS.md
```

## License and citation

The source code, documentation, and supplied `umfnet_whu.pth` checkpoint are distributed under the [MIT License](LICENSE). Publication metadata has not yet been supplied, so no paper citation is included.
