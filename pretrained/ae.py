import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint


def make_group_norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample1= nn.Sequential(
                nn.Conv2d(32,32,kernel_size=4,stride=2,padding=1),
                make_group_norm(32),
                self.activation
        )
        self.downsample2= nn.Sequential(
                nn.Conv2d(32,32,kernel_size=4,stride=2,padding=1),
                make_group_norm(32),
                self.activation
        )
        self.imgconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            make_group_norm(32),
            self.activation
        )

        self.midconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            make_group_norm(32),
            self.activation
        )

        self.out = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.resnet1 = nn.Sequential(*[ResNet(channels=32, activation='leaky') for _ in range(3)])
        self.resnet2 = nn.Sequential(*[ResNet(channels=32, activation='leaky') for _ in range(3)])
        self.resnet3 = nn.Sequential(*[ResNet(channels=32, activation='leaky') for _ in range(3)])

    def forward(self, x): 
        B,C,H,W = x.shape
        h = self.resnet1(self.imgconv(x))
        h = self.downsample1(h) 
        x2 = self.downsample2(self.resnet2(h))
        z = self.resnet3(self.midconv(x2))
        z = self.out(z)
        assert z.shape==torch.Size([B,8,H//4,W//4]), 'encoder error'
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.upsample1=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            make_group_norm(32),
            self.activation
        )
        self.upsample2=nn.Sequential(
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
              make_group_norm(32),
              self.activation
        )
        self.deconv=nn.Sequential(
              nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
              nn.Sigmoid()
        )
        self.input = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            make_group_norm(32),
            self.activation
        )
        self.resnet1 = nn.Sequential(*[ResNet(channels=32, activation='relu') for _ in range(3)])
        self.resnet2 = nn.Sequential(*[ResNet(channels=32, activation='relu') for _ in range(3)])
        self.resnet3 = nn.Sequential(*[ResNet(channels=32, activation='relu') for _ in range(3)])

    def forward(self, z): 
        B,C,H,W = z.shape
        z1 = self.upsample1(self.resnet1(self.input(z)))
        z2 = self.upsample2(self.resnet2(z1))
        z3 = self.deconv(self.resnet3(z2))
        x_recon = z3
        assert x_recon.shape==torch.Size([B, 3, H*4, W*4]), 'recon error'
        return x_recon



class ResNet(nn.Module):
    def __init__(self, channels=3, activation='leaky'):
        super().__init__()

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                make_group_norm(channels),
                self.activation
            ) for _ in range(2)]
        )
    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        out = h+x
        return out
