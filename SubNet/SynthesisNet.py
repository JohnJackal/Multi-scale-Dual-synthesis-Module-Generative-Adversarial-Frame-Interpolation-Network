import torch
import torch.nn as nn
import DenseBlock
import torch.nn.functional as F
from collections import OrderedDict

class SynthesisNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self,
                 growth_rate,
                 num_init_features=12,
                 drop_rate=0.5,
                 bn_size=4,
                 channels=3,
                 ):
        super(SynthesisNet,self).__init__()

        # 1st conv layer (with filter size 5*5)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3,num_init_features,kernel_size=5,stride=1,
                                padding=2,bias=False)),
            ('norm0',nn.BatchNorm2d(num_init_features)),
            ('mish0',DenseBlock.Mish()),
        ]))

        # Each DenseBlock
        