import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from networks.minipointnet import PointNetFeaturePropagation
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from networks.pointnet import PointNetfeat
from common.pointcloud_utils import farthest_point_sample, index_points


class CorrFusion(nn.Module):
    def __init__(self,
                 num_sample,
                 mlp_1,
                 mlp_2,
                #  mlp_pointnet,
                 pointnet_param):
        super(CorrFusion, self).__init__()
        
        self.num_sample = num_sample
        
        self.mlp_maskpred = mlp_1
        self.conv1_mask_list = nn.ModuleList([
            nn.Conv2d(self.mlp_maskpred[i-1], self.mlp_maskpred[i], kernel_size=1, stride=1, padding=0) for i in range(1, len(self.mlp_maskpred))
        ])
        
        self.mlp2_maskpred = mlp_2
        self.conv_mask_list = nn.ModuleList([
            nn.Conv2d(self.mlp2_maskpred[i-1], self.mlp2_maskpred[i], kernel_size=1, stride=1, padding=0) for i in range(1, len(self.mlp2_maskpred))
        ])
        
        # self.pointnet_fp = PointNetFeaturePropagation(
        #                     in_channel=self.mlp2_maskpred[-1], 
        #                     mlp=mlp_pointnet)
        self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        
    def forward(self, pc_1, pc_2, pc_feat_1, pc_feat_2):
        B, N, D = pc_1.size()
        pc_feat_1, pc_feat_2 = pc_feat_1.permute(0, 2, 1), pc_feat_2.permute(0, 2, 1)
        
        pc_2_subsample_idx = farthest_point_sample(pc_2, self.num_sample)
        pc_2_subsample = index_points(pc_2, pc_2_subsample_idx)
        
        feat_product = torch.cat((pc_feat_1.unsqueeze(2).repeat(1, 1, N, 1), pc_feat_2.unsqueeze(1).repeat(1, N, 1, 1)), dim=-1)
        
        feat_product = feat_product.permute(0, 3, 1, 2)

        for i, num_out_channel in enumerate(self.mlp_maskpred[1:]):
            feat_product = self.conv1_mask_list[i](feat_product)
            feat_product = F.relu(feat_product)
        
        feat_product_max = torch.max(feat_product, dim=2, keepdim=True)[0]
        for i, num_out_channel in enumerate(self.mlp2_maskpred[1:-1]):
            feat_product_max = self.conv_mask_list[i](feat_product_max)
            feat_product_max = F.relu(feat_product_max)
            
        feat_flow = self.conv_mask_list[-1](feat_product_max).squeeze(2)
        feat_corr, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1) + feat_flow)
        return feat_flow, feat_corr.permute(0, 2, 1)


class CorrFlowFusion(nn.Module):
    def __init__(self,
                 mlp1_mask,
                 mlp2_mask,
                #  mlp_dist_mask,
                 pointnet_dist_param,
                 pointnet2_param,
                 mlp_flow,
                 pointnet_flow_param):
        super(CorrFlowFusion, self).__init__()
        
        self.mlp_maskpred = mlp1_mask
        self.conv1_mask_list = nn.ModuleList([
            nn.Conv2d(self.mlp_maskpred[i-1], self.mlp_maskpred[i], kernel_size=1, stride=1, padding=0) for i in range(1, len(self.mlp_maskpred))
        ])
        
        self.mlp2_maskpred = mlp2_mask
        self.conv_mask_list = nn.ModuleList([
            nn.Conv2d(self.mlp2_maskpred[i-1], self.mlp2_maskpred[i], kernel_size=1, stride=1, padding=0) for i in range(1, len(self.mlp2_maskpred))
        ])
        
        self.conv_mask = nn.Conv2d(self.mlp_maskpred[-1], 1, kernel_size=1, stride=1, padding=0)
        
        self.conv_prob = nn.Conv2d(self.mlp2_maskpred[-1], 1, kernel_size=1, stride=1, padding=0)

        ### Original network code from part induction
        # self.mlp_dist_mask = mlp_dist_mask
        # self.conv2_dist_mask_list = nn.ModuleList([
        #     nn.Conv2d(self.mlp_dist_mask[i-1], self.mlp_dist_mask[i], kernel_size=1, stride=1, padding=0) for i in range(1, len(self.mlp_dist_mask))
        # ])
        
        self.pointnet_dist_param = pointnet_dist_param
        self.pointnet_dist = PointNetfeat(**pointnet_dist_param)
        
        
        self.pointnet2_param = pointnet2_param
        self.sa1 = PointnetSAModule(**pointnet2_param['sa1'])
        self.sa2 = PointnetSAModule(**pointnet2_param['sa2'])
        
        self.fp_list = nn.ModuleList([])
        for mlp_fp in pointnet2_param['fp']['mlp']:
            self.fp_list.append(PointnetFPModule(mlp=mlp_fp, bn=pointnet2_param['fp']['bn']))

        self.mlp_flow = mlp_flow
        self.conv_flow_list = nn.ModuleList([
            nn.Conv1d(self.mlp_flow[i-1], self.mlp_flow[i], kernel_size=1, stride=1, padding=0) for i in range(1, len(self.mlp_flow))
        ])
        
        self.pc_feature_encoder = PointNetfeat(**pointnet_flow_param)
        
        
    def forward(self, pc_1, pc_2, pc_feat_1, pc_feat_2):
        B, N, D = pc_1.size()
        pc_feat_1, pc_feat_2 = pc_feat_1.permute(0, 2, 1), pc_feat_2.permute(0, 2, 1)
        
        feat_product = torch.cat((pc_feat_1.unsqueeze(2).repeat(1, 1, N, 1), pc_feat_2.unsqueeze(1).repeat(1, N, 1, 1)), dim=-1)
        
        feat_product = feat_product.permute(0, 3, 1, 2)

        for i, num_out_channel in enumerate(self.mlp_maskpred[1:]):
            feat_product = self.conv1_mask_list[i](feat_product)
            feat_product = F.relu(feat_product)
        
        feat_product_max = torch.max(feat_product, dim=2, keepdim=True)[0]
        for i, num_out_channel in enumerate(self.mlp2_maskpred[1:-1]):
            feat_product_max = self.conv_mask_list[i](feat_product_max)
            feat_product_max = F.relu(feat_product_max)
        
        ### Prediction of coarse correspondance mask, B * N * N
        pred_corr_mask_coarse = self.conv_mask(feat_product).squeeze(1)
        
        feat_product_max = self.conv_mask_list[-1](feat_product_max)
        
        ### Prediction of correspondance probability, B * N
        pred_corr_prob = self.conv_prob(feat_product_max).squeeze(1).squeeze(1)
        
        ### Refined correspondance mask, B * N * N
        pred_corr_refined = pred_corr_mask_coarse * pred_corr_prob.unsqueeze(2)
        
        ### Distance matrix between two point cloud, B * C * N * N
        dist_m = pc_2.unsqueeze(1) - pc_1.unsqueeze(2)
        
        ### Distance mask matrix, B * C+1 * N * N
        dist_mask_m = torch.cat([dist_m.permute(0, 3, 1, 2), pred_corr_refined.unsqueeze(1)], dim=1)
        
        ### Original network code from part induction
        ### Feat flow, B * D * N * N
        # for i, num_out_channel in enumerate(self.mlp_dist_mask[1:-1]):
        #     dist_mask_m = self.conv2_dist_mask_list[i](dist_mask_m)
        #     dist_mask_m = F.relu(dist_mask_m)
        # dist_mask_m = self.conv2_dist_mask_list[-1](dist_mask_m)
        
        # ### Correspondance feature after mlp, B * D * N
        # feat_corr = torch.max(dist_mask_m, dim=2)[0].squeeze(2)
        
        ### Pointnet encoder, feat_corr, B * D' * N
        feat_corr = []
        for i in range(N):
            dist_mask_m_i = dist_mask_m[:, :, :, i]
            dist_feat_i_per_point, dist_feat_i, *_ = self.pointnet_dist(dist_mask_m_i)
            feat_corr.append(dist_feat_i)

        feat_corr = torch.stack(feat_corr, dim=2)
        
        
        ### Pointnet2 for flow feature extraction
        l1_xyz, l1_points = pc_1, feat_corr
        l2_xyz, l2_points = self.sa1(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa2(l2_xyz, l2_points)
        
        l2_points = self.fp_list[0](l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp_list[1](l1_xyz, l2_xyz, l1_points, l2_points)
        
        ### Final flow feature, B * D' * N
        l0_points = self.fp_list[2](pc_1, l1_xyz, None, l1_points)

        ### Final MLP to predict flow, B * 3 * N
        for i, num_out_channel in enumerate(self.mlp_flow[1:-1]):
            l0_points = self.conv_flow_list[i](l0_points)
            l0_points = F.relu(l0_points)
        pred_flow = self.conv_flow_list[-1](l0_points)
        
        feat_flow, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1) + pred_flow)
        
        return pred_corr_refined, pred_flow, feat_flow.permute(0, 2, 1)
    