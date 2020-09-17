import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

# define the Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class DenseLayer(nn.Module):    # module isn't a ordered structure.
    def __init__(self,num_input_features,
                 growth_rate,kernel_size,
                 bn_size,drop_rate,
                 memory_efficient=False):
        super(DenseLayer,self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('mish1', Mish())
        self.add_module('conv1',nn.Conv2d(num_input_features,growth_rate * bn_size,
                                          kernel_size=1,stride=1,bias=False))   # the bottleneck layer.
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('mish2', Mish())
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=kernel_size, stride=1, padding = (kernel_size - 1) / 2,bias=False))  # the main conv layer.
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self,input:List(Tensor)) -> Tensor:
        '''
            bottleneck layer function
            type: (List[Tensor]) -> Tensor
        '''
        concated_features = torch.cat(input,1)
        bottleneck_output = self.conv1(self.mish1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self,input:List[Tensor]) -> bool:
        '''
            to detect whether there is any tensor requires grad calculation.
            returns boolean
        '''
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self,input:List[Tensor]) -> Tensor:
        '''
            type: (List[Tensor]) -> Tensor
            perform checkpointed bottleneck.
        '''
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    def forward(self,input):
        # wrap the input to a List[Tensor]
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        # if memory isn't enough then do the checkpointed bn, else do the bn directly.
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        # do the main convolution
        new_features = self.conv2(self.mish2(self.norm2(bottleneck_output)))

        # perform dropout.
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

# define the dense block
class DenseBlock(nn.ModuleDict):    # moduledict is a dict of submodules. (submodules could be called via their names)
    def __init__(self,num_layers,
                 num_input_features,
                 bn_size,growth_rate,
                 drop_rate,
                 memory_efficient=False):
        super(DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,    # densenet's featuremaps would grow with layer number n increases.
                growth_rate = growth_rate,
                bn_size = bn_size,
                drop_rate = drop_rate,
                memory_efficient = memory_efficient,
            )
            self.add_module('denselayer%d' % (i+1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features,1)

# def transition layer class
class TransitionLayer(nn.Sequential):   # Sequential is a ordered structure.
    def __init__(self,num_input_features, num_output_features):
        super(TransitionLayer,self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('mish', Mish())
        self.add_module('conv', nn.Conv2d(num_input_features,num_output_features,
                                          kernel_size=1,stride=1,bias=False))
