from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return Bicubic(args)


class Bicubic(nn.Module):
    def __init__(self, args):
        super(Bicubic, self).__init__()
        self.scale = args.scale[0]
        
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        return x
