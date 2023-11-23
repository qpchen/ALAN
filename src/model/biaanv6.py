import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


def make_model(args, parent=False):
    return BIAANV6(args)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


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


# Attention Branch
class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):

        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):
        
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)


        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class AAB(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=30):
        super(AAB, self).__init__()
        self.t=t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention = AttentionBranch(nf)  
        
        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)         
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False) 

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        attention = self.attention(x)
        non_attention = self.non_attention(x)
    
        x = attention * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out



class BIAANV6(nn.Module):
    
    def __init__(self, args):
        super(BIAANV6, self).__init__()
        
        in_nc  = 3
        out_nc = 3
        nf  = 40
        unf = 24
        nb  = 16
        scale = args.scale[0]

        # AAB
        AAB_block_f = functools.partial(AAB, nf=nf)
        self.scale = scale

        ### define from left to right
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.AAB_trunk = make_layer(AAB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        ### define from right to left
        self.bconv_first = nn.Conv2d(out_nc, unf, 3, 1, 1, bias=True)  # v1:unf; v2:nf

        #### downsampling
        self.bdownconv1 = nn.Conv2d(unf, nf, 3, 1, 1, bias=True)
        self.batt1 = PA(nf)
        self.bLRconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.bdownconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.batt2 = PA(nf)
            self.bLRconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ### main blocks
        self.bAAB_trunk = make_layer(AAB_block_f, nb)
        self.btrunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.bconv_last = nn.Conv2d(nf, in_nc, 3, 1, 1, bias=True)  # v1:nf; v2:unf

    def forward(self, x, y=None):
        if y is None:
            fea = self.conv_first(x)
            trunk = self.trunk_conv(self.AAB_trunk(fea))
            fea = fea + trunk

            if self.scale == 2 or self.scale == 3:
                fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
                fea = self.lrelu(self.att1(fea))
                fea = self.lrelu(self.HRconv1(fea))
            elif self.scale == 4:
                fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
                fea = self.lrelu(self.att1(fea))
                fea = self.lrelu(self.HRconv1(fea))
                fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
                fea = self.lrelu(self.att2(fea))
                fea = self.lrelu(self.HRconv2(fea))

            out = self.conv_last(fea)

            ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
            out = out + ILR

            return out
        else:
            ##############################
            # train from LR to HR
            fea1 = self.conv_first(x)

            trunk = self.trunk_conv(self.AAB_trunk(fea1))
            fea2 = fea1 + trunk

            if self.scale == 2 or self.scale == 3:
                fea3 = self.upconv1(F.interpolate(fea2, scale_factor=self.scale, mode='nearest'))
                fea3 = self.lrelu(self.att1(fea3))
                fea3 = self.lrelu(self.HRconv1(fea3))
            elif self.scale == 4:
                fea3 = self.upconv1(F.interpolate(fea2, scale_factor=2, mode='nearest'))
                fea3 = self.lrelu(self.att1(fea3))
                fea3 = self.lrelu(self.HRconv1(fea3))
                fea3 = self.upconv2(F.interpolate(fea3, scale_factor=2, mode='nearest'))
                fea3 = self.lrelu(self.att2(fea3))
                fea3 = self.lrelu(self.HRconv2(fea3))

            out = self.conv_last(fea3)

            ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
            out = out + ILR

            ##############################
            # train from HR to LR
            bfea1 = self.bconv_first(y)

            if self.scale == 2 or self.scale == 3:
                bfea2 = self.bdownconv1(F.interpolate(bfea1, scale_factor=1/self.scale, mode='nearest'))
                bfea2 = self.lrelu(self.batt1(bfea2))
                bfea2 = self.lrelu(self.bLRconv1(bfea2))
            elif self.scale == 4:
                bfea2 = self.bdownconv1(F.interpolate(bfea1, scale_factor=1/2, mode='nearest'))
                bfea2 = self.lrelu(self.batt1(bfea2))
                bfea2 = self.lrelu(self.bLRconv1(bfea2))
                bfea2 = self.bdownconv2(F.interpolate(bfea2, scale_factor=1/2, mode='nearest'))
                bfea2 = self.lrelu(self.batt2(bfea2))
                bfea2 = self.lrelu(self.bLRconv2(bfea2))

            btrunk = self.btrunk_conv(self.bAAB_trunk(bfea2))
            bfea3 = bfea2 + btrunk

            ################
            # output with features
            bout = self.bconv_last(bfea3)

            bILR = F.interpolate(y, scale_factor=1/self.scale, mode='bilinear', align_corners=False)
            bout = bout + bILR

            # return out, bout
            return out, bout, fea1, fea2, fea3, bfea1, bfea2, bfea3
