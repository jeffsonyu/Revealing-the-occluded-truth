import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    
class WNFdecoder(nn.Module):
    def __init__(self, 
                 grid_dim, 
                 reso_grid,
                 padding,
                 sample_mode,
                 input_dim, 
                 hidden_size,
                 n_blocks):
        super(WNFdecoder, self).__init__()
        
        ### WNF Decoder, inherit from ConvOccNet
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        
        self.grid_dim = grid_dim

        self.fc_c = nn.ModuleList([
            nn.Linear(self.grid_dim, self.hidden_size) for i in range(self.n_blocks)
        ])

        self.fc_p = nn.Linear(self.input_dim, self.hidden_size)
        self.fc_p_img = nn.Linear(self.input_dim+self.grid_dim, self.hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(self.hidden_size) for i in range(self.n_blocks)
        ])

        self.fc_out = nn.Linear(self.hidden_size, 1)
        
    def forward(self, sample_p, sample_p_feat):
        net = self.fc_p(sample_p)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](sample_p_feat)
            net = self.blocks[i](net)

        out = self.fc_out(F.relu(net))
        out = out.squeeze(-1)

        return out