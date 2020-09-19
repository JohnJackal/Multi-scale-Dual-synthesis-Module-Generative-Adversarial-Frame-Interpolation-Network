import torch
import torch.nn as nn
import DenseBlock as DB
import torch.nn.functional as F
from collections import OrderedDict


# define the synthesis subnet
class SynthesisNet(nn.Module):
    r"""
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_num (int) - how many dense blocks there are.
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    def __init__(self,
                 growth_rate=12,
                 num_init_features=12,
                 block_num=16,
                 drop_rate=0.5,
                 bn_size=4,
                 channels=3,
                 ):
        super(SynthesisNet,self).__init__()

        # 1st conv layer (with filter size 5*5)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3,num_init_features, kernel_size=5, stride=1,
                                padding=2, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('mish0', DB.Mish()),
        ]))

        # Each DenseBlock has 3 conv layers, 3 bn layers and 2 activation functions.
        num_features = num_init_features # num of feature maps
        for i in range(block_num):
            block = DB.DenseBlock(
                num_layers=3,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=False
            )
            self.features.add_module('denseblock%d' % (i+1),block)
            # provide input feature map num for next block.
            num_features = num_features + 3 * growth_rate
            # add the transition layer if this isn't the last block.
            if i != block_num-1:
                trans = DB.TransitionLayer(num_input_features=num_features,
                                           num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2

        # Final Conv layer and Mish activation
        self.features.add_module('convout',nn.Conv2d(num_features,1,3,1,1,bias=False))
        self.features.add_module('mishout',DB.Mish())

        # initialization
        # conv layer using kaiming's idea
        # norm layer using norm(0,1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        return features

