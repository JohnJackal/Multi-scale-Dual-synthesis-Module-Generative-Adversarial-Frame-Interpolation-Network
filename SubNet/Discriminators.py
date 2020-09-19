import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# define the frame discriminator
class FDiscriminator(nn.Module):
    def __init__(self,
                 num_layers):   # superior of num_layers is 5.
        super(FDiscriminator,self).__init__()

        mods = OrderedDict()
        in_channels = [3,64,64,64,64]
        out_channels = [64,64,64,64,1]
        # Add conv layers
        mods['conv1'] = nn.Conv2d(in_channels[0], out_channels[0], 3, 2, bias=False)
        for i in range(1,num_layers):
            mods['leakyrelu%d' % (i + 1)] = nn.LeakyReLU(inplace=True)
            mods['conv%d' % (i + 1)] = nn.Conv2d(in_channels[i],out_channels[i],3,2,bias=False)
        self.features = nn.Sequential(mods)

        # initialization
        # conv layers using kaiming's idea
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        features = self.features(x)
        score = torch.sum(features)
        dim = list(features.size())
        num = dim[0]
        for i in range(1, len(dim)):
            num *= dim[i]
        score = F.sigmoid(score / num)
        return score


# define the sequence discriminator
class SDiscriminator(nn.Module):
    def __init__(self):
        super(SDiscriminator,self).__init__()
        mods = OrderedDict()
        in_channels = [3,64,64,64]
        out_channels = [64,64,64,3]
        # Add Conv layers.
        mods['conv1'] = nn.Conv2d(in_channels[0], out_channels[0], 3, 2, bias=False)
        for i in range(1,4):
            mods['leakyrelu%d' % (i + 1)] = nn.LeakyReLU(inplace=True)
            mods['conv%d' % (i + 1)] = nn.Conv2d(in_channels[i], out_channels[i], 3, 2, bias=False)
        self.features = nn.Sequential(mods)

        # initialization
        # conv layers using kaiming's idea
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        features = self.features(x)
        score = torch.sum(features)
        dim = list(features.size())
        num = dim[0]
        for i in range(1, len(dim)):
            num *= dim[i]
        score = F.sigmoid(score / num)
        return score
