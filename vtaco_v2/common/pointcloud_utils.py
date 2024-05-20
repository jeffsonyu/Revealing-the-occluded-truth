import numpy as np
import torch

def norm_pointcloud_np(pc, pc_obj):
    centroid = np.mean(pc_obj, axis=0)
    pc = pc - centroid
    pc_obj = pc_obj - centroid
    
    m = np.max(np.sqrt(np.sum(pc_obj ** 2, axis=1)))
    pc_normalized = pc / (2*m)
    return pc_normalized

def norm_pointcloud(pc, pc_obj):
    centroid = torch.mean(pc_obj, dim=0)
    pc = pc - centroid
    pc_obj = pc_obj - centroid
    
    m = torch.max(torch.sqrt(torch.sum(pc_obj ** 2, dim=1))).to(pc.device)
    pc_normalized = pc / (2*m)
    return pc_normalized


def norm_pointcloud_batch(pc, pc_obj):
    centroid = torch.mean(pc_obj, dim=1, keepdim=True)
    pc = pc - centroid
    pc_obj = pc_obj - centroid
    
    m = torch.max(torch.sqrt(torch.sum(pc_obj ** 2, dim=2)), dim=1, keepdim=True)[0]
    pc_normalized = pc / (2*m.unsqueeze(-1))
    
    return pc_normalized


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def chamfer_distance(pc1, pc2):
    pc1, pc2 = pc1.float(), pc2.float()
    # dist from A to B
    dist_AB = torch.cdist(pc1, pc2, p=2)

    # dist from B to A
    dist_BA = torch.cdist(pc2, pc1, p=2)

    # Chamfer dist
    chamfer_dist = torch.mean(torch.min(dist_AB, dim=1)[0]) + torch.mean(torch.min(dist_BA, dim=1)[0])
    
    return chamfer_dist


def find_points_vec_dist_mask(points, ref_points, threshold):
    """
    Find a mask indicating which points are within a specified distance threshold
    from each reference point for batched input.

    Parameters:
    - points (torch.Tensor): Tensor of shape (B, N, D) where B is the batch size,
      N is the number of points, and D is the dimension.
    - ref_points (torch.Tensor): Tensor of shape (B, M, D) where B is the batch size,
      M is the number of reference points.
    - threshold (float): Distance threshold.

    Returns:
    - torch.Tensor: A boolean mask of shape (B, N, M) where True indicates the point
      is within the threshold of the reference point for each batch.
    """
    # Compute squared distances
    # Utilizing broadcasting: (B, N, 1, D) - (B, 1, M, D) -> (B, N, M, D)
    dist_square = torch.sum((points.unsqueeze(2) - ref_points.unsqueeze(1)) ** 2, dim=3)
    
    dist_vector = points.unsqueeze(2) - ref_points.unsqueeze(1)
    
    # Apply threshold
    within_threshold = dist_square < threshold ** 2
    
    return within_threshold, dist_square, dist_vector



def rotate_hand_points_tensor(hand_points, wrist_pos, device):
    
    def R_from_PYR(wrist_rot):
        roll, pitch, yaw = wrist_rot
        R_roll = np.array([[np.cos(roll), -np.sin(roll), 0],
                        [np.sin(roll), np.cos(roll), 0],
                        [0, 0, 1]])

        R_pitch = np.array([[1, 0, 0],
                            [0, np.cos(pitch), np.sin(pitch)],
                            [0, -np.sin(pitch), np.cos(pitch)]])

        R_yaw = np.array([[np.cos(yaw), 0, -np.sin(yaw)],
                        [0, 1, 0],
                        [np.sin(yaw), 0, np.cos(yaw)]])
        return R_pitch @ R_yaw @ R_roll
    
    rotation_matrix = np.linalg.inv(R_from_PYR(np.array([-np.pi/2, np.pi/2, 0])))
    rotation_matrix_inv = torch.tensor(rotation_matrix, requires_grad=True).float()
    rotation_matrix_inv = rotation_matrix_inv.unsqueeze(0).repeat(hand_points.shape[0], 1, 1)  # Repeat the matrix for each batch
    hand_points_rotate = torch.bmm(rotation_matrix_inv.to(device), hand_points.transpose(1, 2))
    hand_points_rotate = hand_points_rotate.transpose(1, 2) + wrist_pos.reshape(-1, 1, 3)
    return hand_points_rotate