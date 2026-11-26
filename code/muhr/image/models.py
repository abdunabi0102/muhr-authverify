"""Enkoder E_theta, dekoder D_phi (STN bilan) va diskriminator A_psi (2.1-§, 2.1-rasm, 2.1-jadval)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class Encoder(nn.Module):
    """U-Net tipidagi enkoder. Xabar FC orqali G x G o'lchamli fazoviy xaritaga aylantirilib,
    tasvir o'lchamiga interpolyatsiya qilinadi va kirishga biriktiriladi (har bir bit tasvirning
    muayyan sohalariga taqsimlanadi). Chiqishi — qoldiq xarita R_theta, S = C + beta * R_theta."""

    def __init__(self, msg_len=128, base=64, msg_ch=16, grid=8):
        super().__init__()
        self.msg_ch, self.grid = msg_ch, grid
        self.msg_fc = nn.Linear(msg_len, msg_ch * grid * grid)
        self.inc = DoubleConv(3 + msg_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.up1 = DoubleConv(base * 4 + base * 2, base * 2)
        self.up2 = DoubleConv(base * 2 + base, base)
        self.out = nn.Conv2d(base + 3, 3, 1)

    def forward(self, img, msg):
        x = img * 2 - 1
        mm = self.msg_fc(msg * 2 - 1).view(-1, self.msg_ch, self.grid, self.grid)
        mm = F.interpolate(mm, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.inc(torch.cat([x, mm], 1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x2], 1))
        u2 = F.interpolate(u1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, x1], 1))
        return torch.tanh(self.out(torch.cat([u2, x], 1)))


class SpatialTransformer(nn.Module):
    """Affin parametrlarni baholab, tasvirni kanonik holatga qaytaradi (2.3-§, 1-daraja)."""

    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(True),
            nn.Conv2d(16, 32, 5, 2, 2), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(True), nn.Linear(64, 6))
        # identik o'zgartirish bilan initsializatsiya — o'qitish "betaraf" holatdan boshlanadi
        self.loc[-1].weight.data.zero_()
        self.loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.loc(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Decoder(nn.Module):
    """STN (ixtiyoriy) -> 7 ta konvolyutsion blok -> 4x4 qo'pol pulling -> MLP -> msg_len logit.
    Qo'pol pulling bitlarning fazoviy joylashuvini saqlaydi va kichik siljishlarga befarq."""

    def __init__(self, msg_len=128, base=64, use_stn=True, n_blocks=7, pool=4):
        super().__init__()
        self.stn = SpatialTransformer() if use_stn else None
        layers = [conv_bn_relu(3, base)]
        for i in range(n_blocks - 1):
            layers.append(conv_bn_relu(base, base, stride=2 if i in (1, 3, 5) else 1))
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.fc = nn.Sequential(nn.Linear(base * pool * pool, 512), nn.ReLU(inplace=True),
                                nn.Linear(512, msg_len))

    def forward(self, img):
        x = img * 2 - 1
        if self.stn is not None:
            x = self.stn(x)
        return self.fc(self.pool(self.body(x)).flatten(1))   # logitlar


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        c = base // 2
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c * 4, 1))

    def forward(self, img):
        return self.net(img * 2 - 1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
