import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet
from . import Module

class Conv(nn.Module):
    def __init__(self, in_channels,out_channels,up=0):
        super().__init__()
        self.scale_factor = up
        self.up =  nn.Upsample(scale_factor=up, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.scale_factor != 0:
            x1 = self.up(x)
        else:
            x1 = x
        return self.conv(x1)

class  UMFNet(nn.Module):
    block_expansion = resnet.ConvBlock.expansion
    def __init__(self, n_classes=8, sar_dim=1, opt_dim=3, p=0.4, T=8, window_size=3, qkv_dim=8):
        super().__init__()
        self.n_classes = n_classes

        self.relu = nn.ReLU(inplace=True)

        self.SSM = Module.StructureStrengtheningMoudle(window_size=window_size)

        self.conv_sar = nn.Conv2d(sar_dim, 8, kernel_size=1)
        self.conv_opt = nn.Conv2d(opt_dim, 8, kernel_size=1)

        self.optEn = Module.Encoder(in_channels = 8, layers = [3, 4, 6, 3])
        self.sarEn = Module.Encoder(in_channels = 8, layers = [3, 4, 6, 3])

        self.conv_sar1 = nn.Conv2d(64, 8, kernel_size=1)
        self.conv_opt1 = nn.Conv2d(64, 8, kernel_size=1)

        self.fusion4 = Module.ChannelAttentionFusionModule(256 * self.block_expansion, 128 * self.block_expansion, 16, qkv_dim)
        self.fusion3 = Module.ChannelAttentionFusionModule(128 * self.block_expansion, 64 * self.block_expansion, 8, qkv_dim)
        self.fusion2 = Module.ChannelAttentionFusionModule(64 * self.block_expansion, 128, 4, qkv_dim)
        self.fusion1 = Module.ChannelAttentionFusionModule(64, 32, 4, qkv_dim)

        self.fusion0_opt = Module.UncertaintySpatialAttentionFusionModule(8)
        self.fusion0_sar = Module.UncertaintySpatialAttentionFusionModule(8)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.halfconv2 = Conv(128,64)
        self.halfconv1 = Conv(32,16,4)

        self.conv32_feat = Conv(128 * self.block_expansion, 64 * self.block_expansion, 2)
        self.conv32_opt = Conv(128 * self.block_expansion, 64 * self.block_expansion, 2)
        self.conv32_sar = Conv(128 * self.block_expansion, 64 * self.block_expansion, 2)

        self.conv21_feat = Conv(64 * self.block_expansion, 64)
        self.conv21_opt = Conv(64 * self.block_expansion, 64)
        self.conv21_sar = Conv(64 * self.block_expansion, 64)

        self.conv10_feat = Conv(64,16,4)
        self.conv10_opt = Conv(64,8,4)
        self.conv10_sar = Conv(64,8,4)

        self.conv0 = nn.Conv2d(16, 8, kernel_size=1)
        self.Uconv0 = Module.UBlock(ConvBlock=self.conv0, p=p, T=T)

        self.seg = nn.Sequential(
            nn.Conv2d(8 + 8 ,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, n_classes + 1, kernel_size=1)
        )

        self.aux_seg_opt = nn.Sequential(
            nn.Conv2d(8, n_classes + 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes + 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes + 1, n_classes + 1, kernel_size=1)
        )

        self.aux_seg_sar = nn.Sequential(
            nn.Conv2d(8, n_classes + 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes + 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes + 1, n_classes + 1, kernel_size=1)
        )
        self.aux_seg_fused = nn.Conv2d(8, n_classes + 1, kernel_size=1)

    @staticmethod
    @torch.no_grad()
    def branch_disagreement(fused_logits, opt_logits, sar_logits):
        probabilities = [F.softmax(z.float(), dim=1).clamp_min(1e-7)
                         for z in (fused_logits, opt_logits, sar_logits)]
        mean = sum(probabilities) / len(probabilities)
        return (sum((p * (p.log() - mean.log())).sum(1, keepdim=True)
                    for p in probabilities) / len(probabilities)).clamp_min(0.0)

    def compose_eu(self, eu_mc, jsd):
        jsd = F.interpolate(jsd, size=eu_mc.shape[-2:], mode="bilinear", align_corners=False)
        return eu_mc + jsd.expand_as(eu_mc)

    def forward(self, x, y, return_components=False):
        opt = x

        (
            sar, mu, AU, sar_match, gray,
            rho_opt, rho_sar, eta_opt, eta_sar, eta_null, align_conf
        ) = self.SSM(opt, y)

        sar0 = self.conv_sar(sar)
        opt0 = self.conv_opt(opt)

        opt1, opt2, opt3, opt4 = self.optEn(opt0)
        sar1, sar2, sar3, sar4 = self.sarEn(sar0)

        opt0 = self.up4(self.conv_opt1(opt1)) + opt0
        sar0 = self.up4(self.conv_sar1(sar1)) + sar0

        feat3 = self.up2(self.fusion4(opt4, sar4, AU, rho_opt, rho_sar, eta_null, align_conf))
        opt3_fuse = opt3 + feat3
        sar3_fuse = sar3 + feat3

        feat2 = self.up2(self.fusion3(opt3_fuse, sar3_fuse, AU, rho_opt, rho_sar, eta_null, align_conf))
        opt2_fuse = opt2 + feat2
        sar2_fuse = sar2 + feat2

        feat1 = self.halfconv2(self.fusion2(opt2_fuse, sar2_fuse, AU, rho_opt, rho_sar, eta_null, align_conf))
        opt1_fuse = opt1 + feat1
        sar1_fuse = sar1 + feat1

        feat0 = self.halfconv1(self.fusion1(opt1_fuse, sar1_fuse, AU, rho_opt, rho_sar, eta_null, align_conf))

        feat32 =  feat2 + self.conv32_feat(feat3)
        opt32 =  opt2_fuse + self.conv32_opt(opt3_fuse)
        sar32 =  sar2_fuse + self.conv32_sar(sar3_fuse)

        feat21 =  feat1 + self.conv21_feat(feat32)
        opt21 =  opt1_fuse + self.conv21_opt(opt32)
        sar21 =  sar1_fuse + self.conv21_sar(sar32)

        feat_tmp =  feat0 + self.conv10_feat(feat21)
        feat_last, EU_mc = self.Uconv0(feat_tmp)

        aux_out2 = self.aux_seg_opt(opt0)
        aux_out3 = self.aux_seg_sar(sar0)
        aux_fused = self.aux_seg_fused(feat_last)
        jsd = self.branch_disagreement(aux_fused, aux_out2, aux_out3)
        EU = self.compose_eu(EU_mc, jsd)

        opt_last =  opt0 +  self.conv10_opt(opt21)
        sar_last =  sar0 +  self.conv10_sar(sar21)

        out_opt = self.fusion0_opt(opt_last, feat_last, EU)
        out_sar = self.fusion0_sar(sar_last, feat_last, EU)

        out = self.seg(torch.cat([out_opt,out_sar],dim=1))

        AU_list = [
            mu, AU, sar_match, gray,
            rho_opt, rho_sar, eta_opt, eta_sar, eta_null, align_conf
        ]
        result = (out, sar, AU_list, EU, [aux_out2, aux_out3, aux_fused])
        if return_components:
            return (*result, {"EU_mc": EU_mc, "D_JSD": jsd})
        return result
