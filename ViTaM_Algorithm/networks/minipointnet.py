import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pointcloud_utils import square_distance, index_points


class MiniPointNetfeat(nn.Module):
    def __init__(self, nn_channels=(3, 64, 128, 1024)):
        super(MiniPointNetfeat, self).__init__()
        self.nn_channels = nn_channels
        assert len(nn_channels) == 4
        self.conv1 = torch.nn.Conv1d(nn_channels[0], nn_channels[1], 1)
        self.conv2 = torch.nn.Conv1d(nn_channels[1], nn_channels[2], 1)
        self.conv3 = torch.nn.Conv1d(nn_channels[2], nn_channels[3], 1)
        self.bn1 = nn.BatchNorm1d(nn_channels[1])
        self.bn2 = nn.BatchNorm1d(nn_channels[2])
        self.bn3 = nn.BatchNorm1d(nn_channels[3])

    def forward(self, x):
        """
        :param x: (B, C, N) input points
        :return: global feature (B, C') or dense feature (B, C', N)
        """
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # (B, C', 1)
        x = x.view(-1, self.nn_channels[-1])  # (B, C')
        global_feat = x
        x = x.view(-1, self.nn_channels[-1], 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], dim=1), global_feat  # (B, C'+C'', N), (B, C')

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points