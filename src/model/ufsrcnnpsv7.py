from math import sqrt
from model import common

import torch
from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return UFSRCNNPSV7(args)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        i_feats = 2 * n_feats

        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x * self.scale + shortcut

class UFSRCNNPSV7(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor.
    """

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(UFSRCNNPSV7, self).__init__()

        upscale_factor = args.scale[0]
        num_channels = args.n_colors

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map1 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.map2 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.map3 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.map4 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )
        self.map = nn.Sequential(
            self.map1, self.map2, self.map3, self.map4
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Upsample by Deconvolution layer.
        # self.deconv = nn.ConvTranspose2d(56, num_channels, (9, 9), (upscale_factor, upscale_factor),
        #                                  (4, 4), (upscale_factor - 1, upscale_factor - 1))

        # upsample by Pixel shuffle
        conv = common.default_conv
        kernel_size = 3
        self.deconv = nn.Sequential(
            common.Upsampler(conv, upscale_factor, 56, act='prelu'), #'prelu'),
            conv(56, num_channels, kernel_size)
        )

        #############################
        # Bi-direct  training process layers
        # Bi-direct Deconv layer
        self.bideconv = MLP(num_channels)

        # Bi-direct Expanding layer
        self.biexpand = MLP(56)

        # Bi-direct Mapping layer.
        self.bimap1 = MLP(12)
        self.bimap2 = MLP(12)
        self.bimap3 = MLP(12)
        self.bimap4 = MLP(12)
        self.bimap = MLP(12)

        # Bi-direct Shrinking layer
        self.bishrink = MLP(12)

        # Bi-direct Feature extraction layer
        self.bifeature = MLP(56)

        # Initialize model weights.
        # self._initialize_weights()

    def forward(self, x, isTrain = False):
            ############################
            # training from LR to HR
            fea1 = self.feature_extraction(x)
            fea2 = self.shrink(fea1)
            fea3 = self.map1(fea2)
            fea4 = self.map2(fea3)
            fea5 = self.map3(fea4)
            fea6 = self.map4(fea5)
            fea7 = self.expand(fea6)
            out = self.deconv(fea7)

            if isTrain:
                return out, self.bifeature(fea1), self.bishrink(fea2), self.bimap1(fea3), \
                       self.bimap2(fea4), self.bimap3(fea5), self.bimap4(fea6), self.biexpand(fea7)
            else:
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
