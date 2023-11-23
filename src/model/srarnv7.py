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
    return SRARNV7(args)

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
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, deploy=False, layer_norm=False):
        super(ACL, self).__init__()
        self.dwconv = acb.ACBlock(dim, dim, 7, 1, padding=3, groups=dim, deploy=deploy, norm="no") # depthwise AC conv

        if layer_norm:
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
        if hasattr(self, 'norm'):
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
        self.dwconv = acb.ACBlock(dim, dim, 7, 1, padding=3, groups=dim, deploy=deploy, norm="no") # depthwise AC conv

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

def create_head(dim_in, dim_out, layer_norm=True, res_connect="skip", deploy=False):
    if layer_norm:
        if res_connect == "1conv1":
            head = nn.Sequential(
                        LayerNorm(dim_in, eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1))
        elif res_connect == "1acb3":
            head = nn.Sequential(
                        LayerNorm(dim_in, eps=1e-6, data_format="channels_first"),
                        acb.ACBlock(dim_in, dim_out, 3, 1, 1, deploy=deploy, norm="no"))
        elif res_connect == "3acb3":
            head = nn.Sequential(
                        LayerNorm(dim_in, eps=1e-6, data_format="channels_first"),
                        acb.ACBlock(dim_in, dim_in // 4, 3, 1, 1, deploy=deploy, norm="no"),
                        nn.GELU(),
                        nn.Conv2d(dim_in // 4, dim_in // 4, 1, 1, 0),
                        nn.GELU(),
                        acb.ACBlock(dim_in // 4, dim_out, 3, 1, 1, deploy=deploy, norm="no"))
        elif res_connect == "skip":
            if dim_in != dim_out:
                head = nn.Sequential(
                            LayerNorm(dim_in, eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1))
            else:
                head = LayerNorm(dim_in, eps=1e-6, data_format="channels_first")
    else:
        if res_connect == "1conv1":
            head = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1)
        elif res_connect == "1acb3":
            head = acb.ACBlock(dim_in, dim_out, 3, 1, 1, deploy=deploy, norm="no")
        elif res_connect == "3acb3":
            head = nn.Sequential(
                        acb.ACBlock(dim_in, dim_in // 4, 3, 1, 1, deploy=deploy, norm="no"),
                        nn.GELU(),
                        nn.Conv2d(dim_in // 4, dim_in // 4, 1, 1, 0),
                        nn.GELU(),
                        acb.ACBlock(dim_in // 4, dim_out, 3, 1, 1, deploy=deploy, norm="no"))
        elif res_connect == "skip":
            if dim_in != dim_out:
                head = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1)
            else:
                head = None
    return head


########################################
class RACB(nn.Module):
    r"""Residual Asymmetric ConvNeXt Block (RACB), consisting of ACLs. 
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
    def __init__(self, num_layers, dim_in, dim_out, dp_rates, dp_rate_cur, layer_scale_init_value=1e-6,
                    res_connect="1acb3", deploy=False, layer_norm=False):
        super(RACB, self).__init__()
        self.layer = nn.Sequential(
                *[ACL(dim=dim_in, drop_path=dp_rates[dp_rate_cur + j], 
                layer_scale_init_value=layer_scale_init_value, deploy=deploy, layer_norm=layer_norm) 
                for j in range(num_layers)]
        )
        # conv layers for enhancing the translational equivariance 
        self.layer_head = create_head(dim_in, dim_in, layer_norm=layer_norm, res_connect=res_connect, deploy=deploy)
        
        if dim_in != dim_out:
            self.channel_modify = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1)
    def forward(self, x):
        input = x
        x = self.layer(x)
        if self.layer_head is not None:
            x = self.layer_head(x)
        x = x + input
        if hasattr(self, 'channel_modify'):
            x = self.channel_modify(x)
        return x

class SRARNV7(nn.Module):
    """ 
    在V5基础上(相对V6默认保留Pixel Shuffle中的激活层,但也增加参数选择可去掉激活层)
    增加是否在最终输出上合并双三次插值的选项(默认不合并,即不计算插值并和上采样求和)
    并考虑去除主干网络中所有除了ACL(ConvNext block)以外的激活层,但影响似乎不大,未执行
    提供去掉LN层的选项(建议不使用LN,因LN虽然训练会收敛,但不仅开始有影响,收敛趋势也不好)
    同时固定去除acb中的BN(BN对结果影响极大从开始就极差,且完全不收敛)
    Args:
        scale (int): Image magnification factor.
        num_blocks (int): The number of RACB blocks in deep feature extraction.
        upsampling (str): The choice of upsampling method.
    """

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(SRARNV7, self).__init__()

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
        res_connect = args.res_connect
        self.upsampling = args.upsampling
        self.no_bicubic = args.no_bicubic
        use_norm = not args.no_layernorm

        # RGB mean for DIV2K
        if num_channels == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # ##################################################################################
        # Shallow Feature Extraction.
        if use_norm:
            self.shallow = nn.Sequential(
                acb.ACBlock(num_channels, dims[0], 3, 1, 1, deploy=use_inf, norm="no"),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                ,nn.GELU()
            )
        else:
            self.shallow = nn.Sequential(
                acb.ACBlock(num_channels, dims[0], 3, 1, 1, deploy=use_inf, norm="no")
                ,nn.GELU()
            )

        # ##################################################################################
        # Deep Feature Extraction. Must use LayerNorm, or the output just use the bicubic, with PSNR fixed
        self.RACBs = nn.ModuleList()  # Residual Asymmetric ConvNeXt Block (RACB), consisting of ACLs
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_blocks):
            next_i = i if i == self.num_blocks - 1 else i + 1
            block = RACB(depths[i], dims[i], dims[next_i], dp_rates, cur, layer_scale_init_value, res_connect, use_inf, use_norm)
            self.RACBs.append(block)
            cur += depths[i]

        # conv layers for enhancing the translational equivariance 
        # define head of deep feature, input channel dims[-1] to output dims[0]
        # the output receive the residual connect from stem of deep feature
        self.deep_head = create_head(dims[-1], dims[0], layer_norm=use_norm, res_connect=res_connect, deploy=use_inf)

        # ##################################################################################
        # Upsampling
        # if use_norm:
        #     if self.upsampling != 'PixelShuffleDirect' and self.upsampling != 'Deconv':
        #         self.preup = nn.Sequential(
        #             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        #             acb.ACBlock(dims[0], num_up_feat, 3, 1, 1, deploy=use_inf, norm="no")
        #             ,nn.GELU()
        #         )
        #         # if custom the channel setting in upsampling, convert to it
        #         # if num_up_feat != dims[0]:
        #         #     self.upfea = nn.Conv2d(dims[0], num_up_feat, 1, 1, 0)
        # else:
        #     if self.upsampling != 'PixelShuffleDirect' and self.upsampling != 'Deconv':
        #         self.preup = nn.Sequential(
        #             acb.ACBlock(dims[0], num_up_feat, 3, 1, 1, deploy=use_inf, norm="no")
        #             ,nn.GELU()
        #         )
        #         # if custom the channel setting in upsampling, convert to it
        #         # TODO: 搞清楚是不是这里导致b和test模型在去掉norm后，在几十epoch内某时刻，psnr值突然固定某个数完全不变了
        #         # TODO：因为只有这两个规模激活了下面的upfea，也暂时只有这俩有这个psnr固定的现象，或者突然从37降到34上不去
        #         # if num_up_feat != dims[0]:
        #         #     self.upfea = nn.Conv2d(dims[0], num_up_feat, 1, 1, 0)
        if self.upsampling != 'PixelShuffleDirect' and self.upsampling != 'Deconv':
            self.preup = nn.Sequential(
                acb.ACBlock(dims[0], num_up_feat, 3, 1, 1, deploy=use_inf, norm="no")
                ,nn.GELU()
            )
        # Upsampling layer.
        if self.upsampling == 'Deconv':
            # Deconvolution layer.
            self.deconv = nn.ConvTranspose2d(dims[0], num_channels, (9, 9), (self.scale, self.scale),
                                             (4, 4), (self.scale - 1, self.scale - 1))
        elif self.upsampling == 'PixelShuffleDirect':
            acblock = common.default_acb
            self.pixelshuffledirect = common.UpsamplerDirect(acblock, self.scale, dims[0], num_channels, deploy=use_inf, norm="no") #False)
        elif self.upsampling == 'PixelShuffle':
            acblock = common.default_acb
            if args.no_act_ps:
                self.pixelshuffle = nn.Sequential(
                    common.Upsampler(acblock, self.scale, num_up_feat, act=False, deploy=use_inf, norm="no"), #act='gelu' for v5
                    acblock(num_up_feat, num_channels, 3, 1, 1, deploy=use_inf, norm="no")
                )
            else:  # default option
                self.pixelshuffle = nn.Sequential(
                    common.Upsampler(acblock, self.scale, num_up_feat, act='gelu', deploy=use_inf, norm="no"),
                    acblock(num_up_feat, num_channels, 3, 1, 1, deploy=use_inf, norm="no")
                )
        elif self.upsampling == 'Nearest':
            # Nearest + Conv/ACBlock
            if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
                for i in range(int(log(self.scale, 2))):  #  循环 n 次
                    self.add_module(f'up{i}', acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm="no"))
            elif self.scale == 3:  # 缩放因子等于 3
                self.up = acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm="no")
            else:
                # 报错，缩放因子不对
                raise ValueError(f'scale {self.sscale} is not supported. ' 'Supported scales: 2^n and 3.')

            self.postup = nn.Sequential(
                PA(num_up_feat),
                nn.GELU(),
                acb.ACBlock(num_up_feat, num_up_feat, 3, 1, 1, deploy=use_inf, norm="no"),
                nn.GELU(),
                acb.ACBlock(num_up_feat, num_channels, 3, 1, 1, deploy=use_inf, norm="no")
            )

        # Initialize model weights.
        # self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_feature(self, x):
        input = x
        for i in range(self.num_blocks):
            x = self.RACBs[i](x)
        if self.deep_head is not None:
            x = self.deep_head(x)
        x = input + x   # Residual of the whole deep feature Extraction
        return x

    def forward(self, x):
        if hasattr(self, 'sub_mean'):
            x = self.sub_mean(x)

        out = self.shallow(x)

        out = self.forward_feature(out)

        # if hasattr(self, 'upfea'):
        #     out = self.upfea(out)
        
        if self.upsampling == 'Deconv':
            out = self.deconv(out)
        elif self.upsampling == 'PixelShuffleDirect':
            out = self.pixelshuffledirect(out)
        elif self.upsampling == 'PixelShuffle':
            out = self.preup(out)
            out = self.pixelshuffle(out)
        elif self.upsampling == 'Nearest':
            out = self.preup(out)
            if (self.scale & (self.scale - 1)) == 0:  # 缩放因子等于 2^n
                for i in range(int(log(self.scale, 2))):  #  循环 n 次
                    out = getattr(self, f'up{i}')(F.interpolate(out, scale_factor=2, mode='nearest'))
            elif self.scale == 3:  # 缩放因子等于 3
                out = self.up(F.interpolate(out, scale_factor=3, mode='nearest'))
            out = self.postup(out)

        # a: add interpolate
        if not self.no_bicubic:
            out = out + F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        if hasattr(self, 'add_mean'):
            out = self.add_mean(out)

        return out
