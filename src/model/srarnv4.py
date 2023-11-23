from math import sqrt
from math import log
from model import common

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from model import acb

# mainly copied from the fsrcnnacv4 model

def make_model(args, parent=False):
    return SRARNV4(args)

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

# TODO: ACNXBlock

class ACL(nn.Module):
    r""" Asymmetric ConvNeXt Layer (ACL). 
    There are two equivalent implementations of ConvNeXt Block:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    ConvNeXt use (2) as they find it slightly faster in PyTorch. 
    So we use (2) to implement ACL.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, deploy=False):
        super(ACL, self).__init__()
        self.dwconv = acb.ACBlock(dim, dim, 7, 1, padding=3, groups=dim, deploy=deploy) # depthwise AC conv

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ACLV1(nn.Module):
    r""" Asymmetric ConvNeXt Block (ACL) Based on ConvNeXt Block. 
    There are two equivalent implementations of ConvNeXt Block:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    ConvNeXt use (2) as they find it slightly faster in PyTorch. 
    So we use (2) to implement ACL.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, deploy=False):
        super(ACLV1, self).__init__()
        self.dwconv = acb.ACBlock(dim, dim, 7, 1, padding=3, groups=dim, deploy=deploy) # depthwise AC conv

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, 0) # pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

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
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

########################################

class SRARNV4(nn.Module):
    """ 
    在V3基础上给deep特征分析层中每个block增加一个残差连接
    Args:
        scale (int): Image magnification factor.
        num_blocks (int): The number of RACB blocks in deep feature extraction.
        upsampling (str): The choice of upsampling method.
    """

    # TODO: Conv before Residual connect change to 1*3x3conv or 3*3x3conv(save params for too large kernel), same as SwinIR-L setting

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(SRARNV4, self).__init__()

        num_channels = args.n_colors
        # num_feat = args.n_feat
        # num_map_feat = args.n_map_feat
        # num_up_feat = args.n_up_feat
        self.scale = args.scale[0]
        use_inf = args.load_inf

        depths = args.depths
        dims = args.dims
        self.num_blocks = len(depths)
        if args.srarn_up_feat == 0:
            num_up_feat = dims[0]
        else:
            num_up_feat = args.srarn_up_feat
        drop_path_rate = args.drop_path_rate
        layer_scale_init_value = args.layer_init_scale
        self.upsampling = args.upsampling

        # RGB mean for DIV2K
        if num_channels == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # ##################################################################################
        # Shallow Feature Extraction.
        self.shallow = nn.Sequential(
            # nn.Conv2d(num_channels, dims[0], 3, 1, 1),
            acb.ACBlock(num_channels, dims[0], 3, 1, 1, deploy=use_inf),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            ,nn.GELU()  # srarnv3
        )

        # ##################################################################################
        # Deep Feature Extraction.
        self.expand_layers = nn.ModuleList()  # shrink and multiple channel modifying conv layers
        # self.expand_layers.append(shallow)
        for i in range(self.num_blocks - 1):
            expand_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=1, stride=1),
            )
            self.expand_layers.append(expand_layer)

        self.RACBs = nn.ModuleList()  # Residual Asymmetric ConvNeXt Block (RACB), consisting of ACLs
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_blocks):
            RACB = nn.Sequential(
                *[ACL(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, deploy=use_inf) 
                for j in range(depths[i])]
            )
            self.RACBs.append(RACB)
            cur += depths[i]


        # ##################################################################################
        # Upsampling
        if self.upsampling == 'PixelShuffleDirect' or self.upsampling == 'Deconv':
            self.preup = nn.Sequential(
                LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[-1], dims[0], 1, 1, 0)
            )
        else:
            self.preup = nn.Sequential(
                LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
                acb.ACBlock(dims[-1], dims[0], 3, 1, 1, deploy=use_inf),
                nn.GELU()
            )
            # if custom the channel setting in upsampling, convert to it
            if num_up_feat != dims[0]:
                self.upfea = nn.Conv2d(dims[0], num_up_feat, 1, 1, 0)
        
        # Upsampling layer.
        if self.upsampling == 'Deconv':
            # Deconvolution layer.
            self.deconv = nn.ConvTranspose2d(dims[0], num_channels, (9, 9), (self.scale, self.scale),
                                             (4, 4), (self.scale - 1, self.scale - 1))
        elif self.upsampling == 'PixelShuffleDirect':
            acblock = common.default_acb
            self.pixelshuffledirect = common.UpsamplerDirect(acblock, self.scale, dims[0], num_channels, deploy=use_inf) #False)
        elif self.upsampling == 'PixelShuffle':
            acblock = common.default_acb
            self.pixelshuffle = nn.Sequential(
                common.Upsampler(acblock, self.scale, num_up_feat, act='gelu', deploy=use_inf), #False),
                acblock(num_up_feat, num_channels, 3, 1, 1, deploy=use_inf)
            )
        elif self.upsampling == 'Nearest':
            # Nearest + Conv/ACBlock
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
                # nn.PReLU(num_up_feat),
                nn.GELU(),
                acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf),
                # nn.PReLU(num_up_feat),
                nn.GELU(),
                acb.ACBlock(num_up_feat, num_channels, 3, 1, 1, deploy=use_inf)
            )

        # Initialize model weights.
        # self._initialize_weights()

        # self.apply(self._init_weights)


    def _init_weights(self, m):
        # init from ConvNeXt Net:
        # if isinstance(m, (nn.Conv2d, nn.Linear)):  # seems Conv2d in ACBlock have no bias
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        # init from ACBlock's main func (P.S. adding this seems degrade the performance)
        # if isinstance(m, nn.BatchNorm2d):
        #     nn.init.uniform_(m.running_mean, 0, 0.1)
        #     nn.init.uniform_(m.running_var, 0, 0.2)
        #     nn.init.uniform_(m.weight, 0, 0.3)
        #     nn.init.uniform_(m.bias, 0, 0.4)

    # def forward_feature(self, x):
    #     # out = self.feature_extraction(x)
    #     # out = self.ext_act(out)
    #     # out = self.shrink(out)
    #     # out = self.shr_act(out)
    #     for i in range(self.num_blocks):
    #         x = self.expand_layers[i](x)
    #         x = self.RACBs[i](x)
    #     # out = self.expand(out)
    #     # out = self.exp_act(out)
    #     return x

    def forward(self, x):
        if hasattr(self, 'sub_mean'):
            x = self.sub_mean(x)

        # out = self.forward_feature(x)
        out = self.shallow(x)
        s = out
        for i in range(self.num_blocks):
            input = out
            out = self.RACBs[i](out)
            out = input + out   # Residual inside each RACB
            if i < self.num_blocks - 1:
                out = self.expand_layers[i](out)
        
        out = self.preup(out)
        out = out + s   # Residual of the whole deep feature Extraction

        if hasattr(self, 'upfea'):
            out = self.upfea(out)
        
        if self.upsampling == 'Deconv':
            out = self.deconv(out)
        elif self.upsampling == 'PixelShuffle':
            out = self.pixelshuffle(out)
        elif self.upsampling == 'PixelShuffleDirect':
            out = self.pixelshuffledirect(out)
        elif self.upsampling == 'Nearest':
            if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
                for i in range(int(log(self.scale, 2))):  #  循环 n 次
                    out = getattr(self, f'up{i}')(F.interpolate(out, scale_factor=2, mode='nearest'))
            elif self.scale == 3:  # 缩放因子等于 3
                out = self.up(F.interpolate(out, scale_factor=3, mode='nearest'))
            out = self.postup(out)

        # a: add interpolate
        out = out + F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        if hasattr(self, 'add_mean'):
            out = self.add_mean(out)

        return out

    # # The filter weight of each layer is a Gaussian distribution with zero mean and
    # # standard deviation initialized by random extraction 0.001 (deviation is 0).
    # def _initialize_weights(self) -> None:
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
    #             nn.init.zeros_(m.bias.data)

    #     nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
    #     nn.init.zeros_(self.deconv.bias.data)
