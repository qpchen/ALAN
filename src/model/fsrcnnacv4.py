from math import sqrt
from math import log
from model import common

import torch
from torch import nn
import torch.nn.functional as F

from model import acb

def make_model(args, parent=False):
    return FSRCNNACV4(args)

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class FSRCNNACV4(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor.
    """

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(FSRCNNACV4, self).__init__()

        num_channels = args.n_colors
        num_feat = args.n_feat
        num_up_feat = args.n_up_feat
        self.scale = args.scale[0]
        use_inf = args.load_inf

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            acb.ACBlock(num_channels, 56, 5, 1, 2, deploy=use_inf),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            acb.ACBlock(12, 12, 3, 1, 1, deploy=use_inf),
            nn.PReLU(12),
            acb.ACBlock(12, 12, 3, 1, 1, deploy=use_inf),
            nn.PReLU(12),
            acb.ACBlock(12, 12, 3, 1, 1, deploy=use_inf),
            nn.PReLU(12),
            acb.ACBlock(12, 12, 3, 1, 1, deploy=use_inf),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        # self.deconv = nn.ConvTranspose2d(56, num_channels, (9, 9), (self.scale, self.scale),
        #                                  (4, 4), (self.scale - 1, self.scale - 1))

        # Initialize model weights.
        # self._initialize_weights()

        #### IU: upsampling interpolate version
        # self.upconv1 = nn.Sequential(
        #     nn.Conv2d(56, 24, 3, 1, 1, bias=True),
        #     PA(24),
        #     nn.PReLU(24),
        #     nn.Conv2d(24, 24, 3, 1, 1, bias=True),
        #     nn.PReLU(24),
        #     nn.Conv2d(24, num_channels, 3, 1, 1, bias=True)
        # )

        # self.upconv1 = nn.Sequential(
        #     acb.ACBlock(56, 24, 3, 1, 1, deploy=use_inf),
        #     PA(24),
        #     nn.PReLU(24),
        #     acb.ACBlock(24, 24, 3, 1, 1, deploy=use_inf),
        #     nn.PReLU(24),
        #     acb.ACBlock(24, num_channels, 3, 1, 1, deploy=use_inf)
        # )

        self.preup = acb.ACBlock(num_feat, num_up_feat, 3, 1, 1, deploy=use_inf)
        if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
            for i in range(int(log(self.scale, 2))):  #  循环 n 次
                self.add_module(f'up{i}', acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf))
        elif self.scale == 3:  # 缩放因子等于 3
            self.up = acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf)
        else:
            # 报错，缩放因子不对
            raise ValueError(f'scale {self.sscale} is not supported. ' 'Supported scales: 2^n and 3.')

        self.postup = nn.Sequential(
            PA(num_up_feat),
            nn.PReLU(num_up_feat),
            acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf),
            nn.PReLU(num_up_feat),
            acb.ACBlock(num_up_feat, num_channels, 3, 1, 1, deploy=use_inf)
        )

    def forward(self, x):
        # x = self.sub_mean(x)

        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        # out = self.deconv(out)
        out = self.preup(out)
        if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
            for i in range(int(log(self.scale, 2))):  #  循环 n 次
                out = getattr(self, f'up{i}')(F.interpolate(out, scale_factor=2, mode='nearest'))
        elif self.scale == 3:  # 缩放因子等于 3
            out = self.up(F.interpolate(out, scale_factor=3, mode='nearest'))
        out = self.postup(out)

        # a: add interpolate
        out = out + F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        # out = self.add_mean(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)
