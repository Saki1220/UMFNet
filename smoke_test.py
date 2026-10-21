"""Check model inference and loss edge cases using synthetic data."""

import argparse

import numpy as np
import torch

from infer import load_model
from losses import Loss, OhemCrossEntropy
from model import UMFNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if args.checkpoint:
        model, opt_dim, sar_dim = load_model(args.checkpoint, torch.device("cpu"))
    else:
        model, opt_dim, sar_dim = UMFNet(n_classes=7, opt_dim=4).eval(), 4, 1
    with torch.inference_mode():
        result = model(torch.rand(1, opt_dim, 32, 32), torch.rand(1, sar_dim, 32, 32),
                       return_components=True)
    logits, enhanced, au, eu, aux, components = result
    classes = model.n_classes + 1
    assert logits.shape == (1, classes, 32, 32)
    assert enhanced.shape == (1, 1, 32, 32)
    assert eu.shape == (1, 8, 32, 32)
    assert len(au) == 10 and len(aux) == 3
    for tensor in [logits, enhanced, eu, *au, *aux, *components.values()]:
        assert torch.isfinite(tensor).all()

    weight = np.ones(classes, dtype=np.float32)
    ce = OhemCrossEntropy(weight, "cpu")
    pred = torch.randn(1, classes, 8, 8, requires_grad=True)
    ignored = ce(pred, torch.zeros(1, 8, 8, dtype=torch.long))
    assert ignored.item() == 0.0
    ignored.backward()
    assert torch.isfinite(pred.grad).all()
    labels = torch.randint(1, classes, (1, 8, 8))
    predictions = [torch.randn(1, classes, 8, 8, requires_grad=True) for _ in range(4)]
    raw = torch.rand(1, 1, 8, 8)
    enhanced = raw.clone().requires_grad_()
    au = [torch.zeros_like(raw), torch.ones_like(raw), raw, raw]
    au += [torch.full_like(raw, 0.3) for _ in range(6)]
    criterion = Loss(weight, "cpu")
    criterion.set_epoch(10)
    loss = criterion(labels, predictions[0], [raw, enhanced], au, predictions[1:])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictions)
    assert torch.isfinite(enhanced.grad).all()
    print("PASS: model shapes/finiteness, optional checkpoint loading, ignored-label CE, loss backward")


if __name__ == "__main__":
    main()
