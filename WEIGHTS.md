# Checkpoint

Download the checkpoint from the [v1.0.0 Release](https://github.com/Saki1220/UMFNet/releases/tag/v1.0.0):

- [umfnet_whu.pth](https://github.com/Saki1220/UMFNet/releases/download/v1.0.0/umfnet_whu.pth)
- [SHA256SUMS.txt](https://github.com/Saki1220/UMFNet/releases/download/v1.0.0/SHA256SUMS.txt)
- [LICENSE](https://github.com/Saki1220/UMFNet/releases/download/v1.0.0/LICENSE)

| Field | Value |
| --- | --- |
| Filename | `umfnet_whu.pth` |
| Size | 452,189,853 bytes |
| SHA-256 | `f0bf7c15147e363ecfe7b452817a681d1adec03356a8113ae25a55181b7dd0fe` |
| Serialization | Plain PyTorch state dictionary; 1,065 entries |
| Compatible constructor | `UMFNet(n_classes=7, opt_dim=4, sar_dim=1)` |
| License | [MIT](LICENSE), including the supplied checkpoint |

The filename alone does not establish dataset provenance, preprocessing, label semantics, or performance. These details were not present in the supplied source files.

Load with `torch.load(..., weights_only=True)` and `model.load_state_dict(..., strict=True)`. The provided checkpoint was verified with these options.

The binary is distributed as a Release attachment; `.gitignore` excludes checkpoint files from source history. Preserve the MIT license with redistributed copies.
