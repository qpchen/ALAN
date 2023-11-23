from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return BICNNV3(args)


class BICNNV3(nn.Module):
    def __init__(self, args):
        super(BICNNV3, self).__init__()
        num_channels = args.n_colors
        self.scale = args.scale[0]
        self.cconv2 = nn.Conv2d(64, 64, kernel_size=5, padding=5 // 2)
        self.fconv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.fconv3 = nn.Conv2d(64, num_channels, kernel_size=5, padding=5 // 2)
        self.bconv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=5 // 2)
        self.bconv3 = nn.Conv2d(64, num_channels, kernel_size=9, padding=9 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        if y is None:
            # x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
            fea = self.relu(self.fconv1(x))
            fea = self.relu(self.cconv2(fea))
            out = self.fconv3(fea)
            return out
        else:
            # x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
            ffea1 = self.relu(self.fconv1(x))
            ffea2 = self.relu(self.cconv2(ffea1))
            fout = self.fconv3(ffea2)
            bfea1 = self.relu(self.bconv1(y))
            bfea2 = self.relu(self.cconv2(bfea1))
            bout = self.bconv3(bfea2)
            # bout = F.interpolate(bout, scale_factor=1/self.scale, mode='bicubic')
            return fout, bout, ffea1, ffea2, bfea1, bfea2
