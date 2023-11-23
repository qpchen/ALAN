from math import sqrt
from model import common

import torch
from torch import nn
import torch.nn.functional as F

from model import acb

def make_model(args, parent=False):
    return FSRCNNACV3(args)

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

class FSRCNNACV3(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor.
    """

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(FSRCNNACV3, self).__init__()

        num_channels = args.n_colors
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
        self.upconv1 = nn.Conv2d(56, num_channels, 3, 1, 1, bias=True)

        # self.upconv1 = nn.Sequential(
        #     acb.ACBlock(56, 24, 3, 1, 1, deploy=use_inf),
        #     PA(24),
        #     nn.PReLU(24),
        #     acb.ACBlock(24, 24, 3, 1, 1, deploy=use_inf),
        #     nn.PReLU(24),
        #     acb.ACBlock(24, num_channels, 3, 1, 1, deploy=use_inf)
        # )

    def forward(self, x):
        # x = self.sub_mean(x)

        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        # out = self.deconv(out)
        out = self.upconv1(F.interpolate(out, scale_factor=self.scale, mode='nearest'))

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
