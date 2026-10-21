"""Run UMFNet on a prepared, co-registered optical/SAR NumPy pair."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import UMFNet


def load_input(path, layout):
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        raise ValueError(f"{path}: expected one 3D .npy array")
    if layout == "hwc":
        array = array.transpose(2, 0, 1)
    if min(array.shape) < 1 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path}: supply a non-empty floating-point array with training preprocessing applied")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: input contains non-finite values")
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a plain UMFNet state_dict checkpoint")
    try:
        opt_dim = state["conv_opt.weight"].shape[1]
        sar_dim = state["conv_sar.weight"].shape[1]
        output_classes = state["seg.3.weight"].shape[0]
    except (KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Expected a plain, unprefixed UMFNet state_dict checkpoint") from exc
    if sar_dim != 1 or output_classes < 2:
        raise ValueError("This implementation requires one SAR channel and at least two output classes")
    model = UMFNet(n_classes=output_classes - 1, opt_dim=opt_dim, sar_dim=sar_dim)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), opt_dim, sar_dim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opt", type=Path, required=True, help="Prepared optical .npy array")
    parser.add_argument("--sar", type=Path, required=True, help="Prepared SAR .npy array")
    parser.add_argument("--layout", choices=("chw", "hwc"), default="chw")
    parser.add_argument("--output", type=Path, required=True, help="Output class-index .npy mask")
    parser.add_argument("--details", type=Path, help="Optional .npz with logits, AU, EU, EU_mc and D_JSD")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("--output must end in .npy")
    if args.details and args.details.suffix.lower() != ".npz":
        parser.error("--details must end in .npz")
    torch.manual_seed(args.seed)
    opt, sar = load_input(args.opt, args.layout), load_input(args.sar, args.layout)
    model, opt_dim, sar_dim = load_model(args.checkpoint, torch.device(args.device))
    if opt.shape[1] != opt_dim or sar.shape[1] != sar_dim:
        raise ValueError(f"Checkpoint requires optical={opt_dim}, SAR={sar_dim} channels")
    if opt.shape[-2:] != sar.shape[-2:]:
        raise ValueError("Optical and SAR spatial dimensions must match and be co-registered")
    height, width = opt.shape[-2:]
    padding = (0, (-width) % 32, 0, (-height) % 32)
    opt = F.pad(opt, padding, mode="replicate").to(args.device)
    sar = F.pad(sar, padding, mode="replicate").to(args.device)
    with torch.inference_mode():
        logits, _, au, eu, _, components = model(opt, sar, return_components=True)
    if not torch.isfinite(logits).all() or not torch.isfinite(eu).all():
        raise ValueError("Non-finite model output; check inputs and checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction = logits.argmax(1)[0, :height, :width].cpu().numpy().astype(np.int64)
    np.save(args.output, prediction, allow_pickle=False)
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        values = {"logits": logits, "AU": au[1], "EU": eu, **components}
        np.savez_compressed(args.details, **{
            key: value[0, :, :height, :width].cpu().numpy() for key, value in values.items()
        })
    print(f"Saved {args.output} ({height} x {width}, {logits.shape[1]} output classes)")


if __name__ == "__main__":
    main()
