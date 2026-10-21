import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import resnet

class Encoder(nn.Module):
    block_expansion = resnet.ConvBlock.expansion
    def __init__(self, in_channels=3, layers = [3, 4, 6, 3]):
        super().__init__()

        self.conv_channel = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet = resnet.resnet_custom(64, layers = layers)

        self.up4 = self._upsample_block(512 * self.block_expansion, 256 * self.block_expansion)
        self.up3 = self._upsample_block(256 * self.block_expansion, 128 * self.block_expansion)
        self.up2 = self._upsample_block(128 * self.block_expansion, 64 * self.block_expansion)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64 * self.block_expansion, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.conv_channel(x)
        x1, x2, x3, x4 = self.resnet(x0)

        d4 = self.up4(x4) + x3
        d3 = self.up3(d4) + x2
        d2 = self.up2(d3) + x1
        d1= self.conv1(d2) + x0
        return d1, d2, d3, d4

class UBlock(nn.Module):
    def __init__(self, ConvBlock, p=0.5, T=8):
        super().__init__()
        self.p = p
        self.conv = ConvBlock
        self.T = T

    def Channel_Mask(self, x, p):
        B, C, H, W = x.shape
        mask = (torch.rand(C, device=x.device) > p).float()
        mask = mask.view(1, C, 1, 1)
        return x * mask

    def forward(self, x):

        out = self.conv(x)

        with torch.no_grad():
            mean = torch.zeros_like(out)
            m2 = torch.zeros_like(out)

            for t in range(self.T):
                out_t = self.conv(self.Channel_Mask(x, self.p))
                if t == 0:
                    mean = out_t

                else:
                    delta = out_t - mean
                    mean = mean + delta / (t + 1)
                    m2 = m2 + delta * (out_t - mean)

            if self.T > 1:
                EU = m2 / (self.T - 1)
            else:
                EU = torch.zeros_like(out)

        return out, EU

    def sample(self, x):
        return self.conv(self.Channel_Mask(x, self.p))

class ChannelAttentionFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels=None, down=8, qkv_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.qkv_dim = qkv_dim

        self.conv_au = nn.Conv2d(1, in_channels, kernel_size=3, stride=down, padding=1)
        self.conv_feat = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        kernel_size = int(abs((math.log(in_channels, 2) + 1) / 2))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.eca_conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

        self.mlp_opt = self._build_mlp(in_channels)
        self.mlp_sar = self._build_mlp(in_channels)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.fuse_conv = nn.Conv2d(in_channels * 2, self.out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.gate_scale = nn.Parameter(torch.tensor(-2.0))

        self.q_proj = nn.Linear(1, qkv_dim, bias=False)
        self.k_proj = nn.Linear(1, qkv_dim, bias=False)
        self.v_proj = nn.Linear(1, qkv_dim, bias=False)
        self.out_proj = nn.Linear(qkv_dim, 1, bias=False)
        self.qkv_norm = nn.LayerNorm(in_channels)

    def _build_mlp(self, channels, reduction=4):
        hidden = max(4, channels // reduction)
        return nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )

    def QKV_fusion(self, feat_des, au_des):
        b, c = feat_des.size(0), feat_des.size(1)
        feat = feat_des.view(b, c, 1)
        au = au_des.view(b, c, 1)

        q = self.q_proj(feat)
        k = self.k_proj(au)
        v = self.v_proj(feat)

        evidence_kernel = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.qkv_dim))
        out = self.out_proj(evidence_kernel * v).squeeze(-1)
        out = self.qkv_norm(out + feat.squeeze(-1))
        return out.view(b, c, 1, 1)

    def _pool_reliability(self, rho, size):
        if rho is None:
            return None
        if rho.shape[-2:] != size:
            rho = F.interpolate(rho, size=size, mode="bilinear", align_corners=False)
        return self.avg_pool(rho.clamp(1e-4, 1.0 - 1e-4)).flatten(1)

    def _pool_source_conf(self, eta_null, align_conf, size):
        source_conf = None
        if eta_null is not None:
            if eta_null.shape[-2:] != size:
                eta_null = F.interpolate(eta_null, size=size, mode="bilinear", align_corners=False)
            source_conf = 1.0 - self.avg_pool(eta_null.clamp(0.0, 1.0)).flatten(1)
        if align_conf is not None:
            if align_conf.shape[-2:] != size:
                align_conf = F.interpolate(align_conf, size=size, mode="bilinear", align_corners=False)
            align_pool = self.avg_pool(align_conf.clamp(1e-4, 1.0)).flatten(1)
            source_conf = align_pool if source_conf is None else source_conf * align_pool
        return source_conf

    def forward(self, opt, sar, au, rho_opt=None, rho_sar=None, eta_null=None, align_conf=None):
        feat = self.conv_feat(torch.cat([opt, sar], dim=1))
        feat_au = self.conv_au(au)

        avg_au = self.avg_pool(feat_au)
        max_au = self.max_pool(feat_au)
        au_des = avg_au + max_au

        avg_feat = self.avg_pool(feat)
        max_feat = self.max_pool(feat)
        feat_des = avg_feat + max_feat

        y = self.QKV_fusion(feat_des, au_des)

        y1 = y.squeeze(-1).transpose(-1, -2)
        y1 = self.eca_conv(y1)
        att_unc = self.sigmoid(y1).transpose(-1, -2).squeeze(-1)

        rel_opt = self._pool_reliability(rho_opt, opt.shape[-2:])
        rel_sar = self._pool_reliability(rho_sar, opt.shape[-2:])
        rel_bias = 0.0
        if rel_opt is not None and rel_sar is not None:
            rel_bias = torch.log(rel_opt + 1e-4) - torch.log(rel_sar + 1e-4)

        source_conf = self._pool_source_conf(eta_null, align_conf, opt.shape[-2:])
        conf_bias = 0.0
        null_gate = None
        if source_conf is not None:
            source_conf = source_conf.clamp(1e-4, 1.0 - 1e-4)
            conf_bias = 0.5 * (torch.log(source_conf) - torch.log(1.0 - source_conf))
            null_gate = 1.0 - source_conf.mean(dim=1, keepdim=True).view(-1, 1, 1, 1)

        gate_scale = torch.sigmoid(self.gate_scale)
        g_opt = self.sigmoid(gate_scale * (self.mlp_opt(att_unc) + rel_bias + conf_bias)).unsqueeze(2).unsqueeze(3)
        g_sar = self.sigmoid(gate_scale * (self.mlp_sar(att_unc) - rel_bias + conf_bias)).unsqueeze(2).unsqueeze(3)

        opt_w = opt * g_opt
        sar_w = sar * g_sar
        fused_out = self.fuse_conv(torch.cat([opt_w, sar_w], dim=1))
        neutral = self.conv1(feat)
        if null_gate is not None:
            fused_out = (1.0 - null_gate) * fused_out

        return self.relu(fused_out + neutral)

class DualSobel(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]]).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag, gx, gy

class StructureStrengtheningMoudle(nn.Module):
    epsilon = 1e-6

    def __init__(self, window_size=3, base_channels=32, align_radius=1):
        super().__init__()
        self.window_size = window_size
        self.align_radius = align_radius
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)
        self.sobel = DualSobel()
        self.tau_evidence = 0.5

        self.au_head = nn.Sequential(
            nn.Conv2d(12, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 4, kernel_size=1)
        )

        self.src_head = nn.Sequential(
            nn.Conv2d(11, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 3, kernel_size=1)
        )

        self.phi_opt = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        self.phi_sar = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        self.rel = nn.Sequential(
            nn.Conv2d(base_channels * 4 + 5, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, groups=base_channels, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=1)
        )
        self.ref_head = nn.Sequential(
            nn.Conv2d(11, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1)
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _gray(self, opt):
        if opt.size(1) >= 3:
            weight = opt.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            return (opt[:, :3] * weight).sum(1, keepdim=True)
        return opt[:, :1]

    def _local_var(self, x):
        mean = self.pool(x)
        var = (self.pool(x * x) - mean * mean).clamp_min(0.0)
        return mean, var

    def _lee_filter(self, sar):
        mean, var = self._local_var(sar)
        noise_var = var.mean(dim=(2, 3), keepdim=True)
        weight = var / (var + noise_var + self.epsilon)
        return mean + weight * (sar - mean)

    def _contrast(self, x):
        mean, var = self._local_var(x)
        return (x - mean).abs() / torch.sqrt(var + self.epsilon)

    def _coherence(self, gx, gy):
        jxx = self.pool(gx * gx)
        jyy = self.pool(gy * gy)
        jxy = self.pool(gx * gy)
        delta = torch.sqrt((jxx - jyy).pow(2) + 4.0 * jxy.pow(2) + self.epsilon)
        return delta / (jxx + jyy + self.epsilon)

    def _observation_priors(self, opt, gray, e_opt, e_sar, c_opt, c_sar, align_conf):
        if opt.size(1) >= 3:
            rgb = opt[:, :3]
        else:
            rgb = gray.repeat(1, 3, 1, 1)

        brightness = rgb.mean(dim=1, keepdim=True)
        chroma = rgb.std(dim=1, keepdim=True, unbiased=False)

        cloud_risk = torch.sigmoid(16.0 * (brightness - 0.72)) * torch.sigmoid(16.0 * (0.20 - chroma))
        shadow_risk = torch.sigmoid(16.0 * (0.18 - brightness)) * torch.sigmoid(10.0 * (0.28 - c_opt))
        opt_flat_risk = torch.sigmoid(12.0 * (0.06 - c_opt)) * (1.0 - cloud_risk)
        sar_flat_risk = torch.sigmoid(12.0 * (0.06 - c_sar)) * torch.sigmoid(16.0 * (0.02 - e_sar))

        structure_gap = (e_sar - e_opt).abs() / (e_sar + e_opt + self.epsilon)
        alignment_risk = (1.0 - align_conf.clamp(0.0, 1.0)) * structure_gap.clamp(0.0, 1.0)

        opt_noise = 1.0 - (1.0 - cloud_risk) * (1.0 - shadow_risk) * (1.0 - opt_flat_risk)
        sar_noise = sar_flat_risk
        sensor_noise = 1.0 - (1.0 - opt_noise) * (1.0 - sar_noise)
        joint_noise = 1.0 - (1.0 - sensor_noise) * (1.0 - 0.35 * alignment_risk)

        opt_reliability = (1.0 - opt_noise) * (0.65 + 0.35 * align_conf.clamp(0.0, 1.0))
        sar_reliability = (1.0 - sar_noise) * (0.45 + 0.55 * c_sar.clamp(0.0, 1.0))
        null_risk = 1.0 - (1.0 - sensor_noise) * (1.0 - 0.75 * alignment_risk)

        return (
            joint_noise.clamp(0.0, 1.0),
            opt_reliability.clamp(0.02, 1.0),
            sar_reliability.clamp(0.02, 1.0),
            null_risk.clamp(0.0, 1.0),
        )

    def _shift(self, x, dy, dx):
        r = self.align_radius
        if r == 0:
            return x
        _, _, h, w = x.shape
        padded = F.pad(x, (r, r, r, r), mode="replicate")
        y0 = r + dy
        x0 = r + dx
        return padded[:, :, y0:y0 + h, x0:x0 + w]

    def _align_opt_edge(self, e_opt, e_sar):
        shifts = []
        scores = []
        for dy in range(-self.align_radius, self.align_radius + 1):
            for dx in range(-self.align_radius, self.align_radius + 1):
                shifted = self._shift(e_opt, dy, dx)
                num = self.pool(shifted * e_sar)
                den = torch.sqrt(self.pool(shifted * shifted) * self.pool(e_sar * e_sar) + self.epsilon)
                shifts.append(shifted)
                scores.append(num / den)
        score = torch.cat(scores, dim=1)
        pi = torch.softmax(score / 0.2, dim=1)
        stacked = torch.stack(shifts, dim=1)
        aligned = (pi.unsqueeze(2) * stacked).sum(dim=1)
        align_conf = pi.max(dim=1, keepdim=True).values
        return aligned, align_conf

    def forward(self, opt, sar):
        gray = self._gray(opt)
        sar_f = self._lee_filter(sar)

        e_opt, gx_o, gy_o = self.sobel(gray)
        e_sar, gx_s, gy_s = self.sobel(sar_f)
        e_opt_a, align_conf = self._align_opt_edge(e_opt, e_sar)

        k_opt = self._coherence(gx_o, gy_o)
        k_sar = self._coherence(gx_s, gy_s)
        c_opt = self._contrast(gray).clamp(0.0, 6.0) / 6.0
        c_sar = self._contrast(sar_f).clamp(0.0, 6.0) / 6.0
        au_prior, opt_rel_prior, sar_rel_prior, null_prior = self._observation_priors(
            opt, gray, e_opt_a, e_sar, c_opt, c_sar, align_conf
        )

        mean_opt, var_opt = self._local_var(gray)
        mean_sar, var_sar = self._local_var(sar)
        log_opt = torch.log1p(255.0 * mean_opt.clamp_min(0.0)) / math.log(256.0)
        log_sar = torch.log1p(255.0 * mean_sar.clamp_min(0.0)) / math.log(256.0)

        residual = torch.cat([
            (log_sar - log_opt).abs(),
            (e_sar - e_opt_a).abs(),
            var_opt,
            var_sar,
            c_opt,
            c_sar,
            k_opt,
            k_sar
        ], dim=1)
        au_in = torch.cat([log_opt, log_sar, e_opt_a, e_sar, residual], dim=1)
        mu_a, sigma_raw, e_opt_raw, e_sar_raw = self.au_head(au_in).chunk(4, dim=1)
        sigma_learned = F.softplus(sigma_raw) + self.epsilon
        sigma_a = 0.5 * sigma_learned + torch.log1p(6.0 * au_prior) + 0.05
        evidence = F.softplus(torch.cat([e_opt_raw, e_sar_raw], dim=1)) + self.epsilon
        evidence = evidence * torch.cat([opt_rel_prior, sar_rel_prior], dim=1)
        evidence_sum = evidence.sum(dim=1, keepdim=True)
        rho_opt = evidence[:, 0:1] / (evidence_sum + self.tau_evidence)
        rho_sar = evidence[:, 1:2] / (evidence_sum + self.tau_evidence)

        src_in = torch.cat([
            sigma_a, rho_opt, rho_sar, e_opt_a, e_sar, (e_opt_a - e_sar).abs(),
            k_opt, k_sar, c_opt, c_sar, align_conf
        ], dim=1)
        src_alpha = F.softplus(self.src_head(src_in)) + self.epsilon
        src_prior = torch.cat([opt_rel_prior, sar_rel_prior, 0.25 + 1.75 * null_prior], dim=1)
        src_alpha = src_alpha * src_prior.clamp(0.02, 2.0)
        src_sum = src_alpha.sum(dim=1, keepdim=True)
        eta_opt = src_alpha[:, 0:1] / src_sum
        eta_sar = src_alpha[:, 1:2] / src_sum
        eta_null = src_alpha[:, 2:3] / src_sum

        h_opt = self.phi_opt(torch.cat([gray, e_opt_a, k_opt], dim=1))
        h_sar = self.phi_sar(torch.cat([sar, e_sar, k_sar], dim=1))
        h_rel = torch.cat([
            eta_opt * h_opt,
            eta_sar * h_sar,
            h_opt * h_sar,
            (h_opt - h_sar).abs(),
            sigma_a,
            rho_opt,
            rho_sar,
            eta_null,
            align_conf
        ], dim=1)
        residual_s = self.rel(h_rel)
        refine_gate = (1.0 - eta_null) * torch.sigmoid(self.ref_head(src_in))
        sar_enh = sar + torch.tanh(self.alpha) * refine_gate * residual_s

        return sar_enh, mu_a, sigma_a, log_sar, log_opt, rho_opt, rho_sar, eta_opt, eta_sar, eta_null, align_conf

class UncertaintySpatialAttentionFusionModule(nn.Module):
    def __init__(self,in_channels ,kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv_opt = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_fusion = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def spatial_attention(self, x, conv):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = conv(x_cat)
        return self.sigmoid(attn)

    def forward(self, x1, x2, unc):
        spatial_map_opt = self.spatial_attention(unc, self.conv_opt)
        spatial_map_fusion = self.spatial_attention(unc, self.conv_fusion)

        x1_w = x1 * spatial_map_opt
        x2_w = x2 * spatial_map_fusion
        out1 = x1 + x2 + x1 * x2
        out2 = x1_w + x2_w + x1_w * x2_w
        out  = out1 + out2 + out1 * out2
        return out
