import torch.nn.functional as F
import torch
import torch.nn as nn


class OhemCrossEntropy(nn.Module):
    def __init__(self, weight, device, ignore_label = 0 , thresh = 0.7, aux_weights = [1, 1]):
        super().__init__()
        self.name = "OhemCrossEntropy"
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float, device=device))

        self.weight = torch.from_numpy(weight).float().to(device)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        valid = labels.view(-1) != self.ignore_label
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            return preds.sum() * 0.0

        n_min = max(valid_count // 16, 1)
        loss = self.criterion(preds, labels).view(-1)[valid]
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(min(n_min, loss.numel()))
        loss1 = torch.mean(loss_hard)
        return loss1

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

class Loss(nn.Module):
    def __init__(
        self,
        weight,
        device,
        ignore_label=0,
        au_weight=0.005,
        ssm_weight=0.02,
        align_weight=0.005,
        uncertainty_start_epoch=3,
        uncertainty_warmup_epochs=5,
        fused_aux_weight=0.3,
    ):
        super().__init__()
        self.CEloss = OhemCrossEntropy(weight=weight, device=device, ignore_label = ignore_label , thresh = 0.7, aux_weights = [1, 1])
        self.MSEloss = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.current_epoch = 1
        self.warmup_epochs = max(1, int(uncertainty_warmup_epochs))
        self.uncertainty_start_epoch = max(1, int(uncertainty_start_epoch))
        self.au_weight = float(au_weight)
        self.ssm_weight = float(ssm_weight)
        self.align_weight = float(align_weight)
        self.fused_aux_weight = float(fused_aux_weight)
        self.name = "Loss"

    def set_epoch(self, epoch, warmup_epochs=None, uncertainty_start_epoch=None):
        self.current_epoch = max(1, int(epoch))
        if warmup_epochs is not None:
            self.warmup_epochs = max(1, int(warmup_epochs))
        if uncertainty_start_epoch is not None:
            self.uncertainty_start_epoch = max(1, int(uncertainty_start_epoch))

    def _uncertainty_ramp(self):
        if self.current_epoch < self.uncertainty_start_epoch:
            return 0.0
        progress = self.current_epoch - self.uncertainty_start_epoch + 1
        return min(1.0, float(progress) / float(self.warmup_epochs))
    
    def _grad_mag(self, x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, lbl, out, sar_list, AU_list, aux ):
        sar, sar_enhance = sar_list
        mu, AU, log_sar, log_opt = AU_list[:4]
        rho_opt = rho_sar = eta_opt = eta_sar = eta_null = align_conf = None
        if len(AU_list) >= 10:
            rho_opt, rho_sar, eta_opt, eta_sar, eta_null, align_conf = AU_list[4:10]
        aux_out2, aux_out3, aux_fused = aux
        loss_main = self.CEloss(out, lbl)
        loss_aux2 = self.CEloss(aux_out2, lbl)      
        loss_aux3 = self.CEloss(aux_out3, lbl)        
        loss_CE = (loss_main + 0.5 * loss_aux2 + 0.3 * loss_aux3
                   + self.fused_aux_weight * self.CEloss(aux_fused, lbl))

        residual = (log_sar - log_opt - mu).abs()
        loss_AU = torch.mean(residual / (AU + 1e-6) + torch.log(AU + 1e-6))

        loss_rad = self.smooth_l1(self.pool(sar_enhance), self.pool(sar))
        loss_tv = torch.mean(torch.abs(sar_enhance[:, :, :, 1:] - sar_enhance[:, :, :, :-1])) + \
                  torch.mean(torch.abs(sar_enhance[:, :, 1:, :] - sar_enhance[:, :, :-1, :]))
        loss_SSM = loss_rad + 0.05 * loss_tv

        if eta_opt is not None and eta_sar is not None and eta_null is not None:
            edge_enh = self._grad_mag(sar_enhance)
            edge_sar = self._grad_mag(sar).detach()
            loss_edge = torch.mean(eta_sar.detach() * torch.abs(edge_enh - edge_sar))
            loss_null = torch.mean(eta_null.detach() * torch.abs(sar_enhance - sar))
            loss_SSM = loss_SSM + 0.05 * loss_edge + 0.05 * loss_null

        loss_align = 0.0
        if eta_opt is not None and align_conf is not None:
            loss_align = torch.mean(eta_opt * (1.0 - align_conf))

        ramp = self._uncertainty_ramp()
        loss = loss_CE + ramp * (
            self.au_weight * loss_AU
            + self.ssm_weight * loss_SSM
            + self.align_weight * loss_align
        )
        return loss
    
