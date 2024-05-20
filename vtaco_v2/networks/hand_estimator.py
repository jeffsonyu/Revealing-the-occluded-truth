import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.minipointnet import MiniPointNetfeat
from networks.pointnet import PointNetfeat
from networks.transformer import TransformerFusion
from components.manopth.manopth.manolayer_old import ManoLayer

class MLPDecoder(nn.Module):
    def __init__(self, nn_channels):
        super(MLPDecoder, self).__init__()
        
        self.nn_channels = nn_channels
        self.fc_list = nn.ModuleList([
            nn.Linear(self.nn_channels[i], self.nn_channels[i+1]) for i in range(len(self.nn_channels)-1)
        ])
        

    def forward(self, x):
        for i in range(len(self.nn_channels)-2):
            x = F.relu(self.fc_list[i](x))
        
        x = self.fc_list[-1](x)
        
        return x

class HandTracker(nn.Module):
    def __init__(self, pointnet_param, mlp_param, manolayer_param):
        super(HandTracker, self).__init__()
        
        ### Pointcloud feature extractor
        self.pointnet = PointNetfeat(**pointnet_param)
        
        ### Pose param decoder
        self.MLP = MLPDecoder(**mlp_param)
        
        ### Manolayer
        self.manolayer = ManoLayer(**manolayer_param)
        
    def forward(self, pc_1, pc_2):
        x_feat, x_feat_global, *_ = self.pointnet(pc_2.permute(0, 2, 1))
        mano_out = self.MLP(x_feat_global)
        
        return mano_out


class HandTracker_v2(nn.Module):
    def __init__(self, pointnet_param, transformer_param, mlp_param, manolayer_param):
        super(HandTracker_v2, self).__init__()
        
        ### Pointcloud feature extractor
        self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        
        ### Transformer Fusion
        self.transformerfuser = TransformerFusion(**transformer_param)
        
        ### Pose param decoder
        self.MLP = MLPDecoder(**mlp_param)
        
        ### Manolayer
        self.manolayer = ManoLayer(**manolayer_param)
        
    def forward(self, pc_1, pc_2):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Transformer Fusion feature from two pc
        pc_feat_fused = self.transformerfuser(pc_2_feat, pc_2, pc_1_feat, pc_1)
        
        pc_feat_final = torch.max(pc_feat_fused, dim=1)[0]
        
        ### Pose decoder
        mano_out = self.MLP(pc_feat_final)
        
        return mano_out