from torch import nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return BISRCNN(args)

class FSRCNN(nn.Module):
    def __init__(self, args, convh):
        super(FSRCNN, self).__init__()
        num_channels = args.n_colors
        self.scale = args.scale[0]
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = convh
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        out = self.conv3(fea)
        return out

class BSRCNN(nn.Module):
    def __init__(self, args, convh):
        super(BSRCNN, self).__init__()
        num_channels = args.n_colors
        self.scale = args.scale[0]
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = convh
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=1/self.scale, mode='bicubic')
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        out = self.conv3(fea)
        # out = F.interpolate(out, scale_factor=1/self.scale, mode='bicubic')
        return out

class BISRCNN(nn.Module):
    def __init__(self, args):
        super(BISRCNN, self).__init__()
        num_channels = args.n_colors
        self.scale = args.scale[0]
        self.convc = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.fsrcn = FSRCNN(args, self.convc)
        self.bsrcn = BSRCNN(args, self.convc)

    def forward(self, x, y=None):
        if y is None:
            return self.fsrcn(x)
        else:
            outf = self.fsrcn(x)
            outb = self.bsrcn(y)
            return outf, outb
