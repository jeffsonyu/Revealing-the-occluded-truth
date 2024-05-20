import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import pytorch_lightning as pl

def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')
    

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class BasicBlockBase(pl.LightningModule):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 D=3):
        super(BasicBlockBase, self).__init__()
        self.save_hyperparameters()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
    if norm_type == 'BN':
        return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    elif norm_type == 'IN':
        return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')
