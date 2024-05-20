import torch

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

# Example usage
B, N, M, D = 1, 10000, 23, 3  # Small numbers for easy verification
points = torch.randn(B, N, D, requires_grad=True) # Obj verts
ref_points = torch.randn(B, M, D, requires_grad=True) # Anchor points, computed by the mano pose and original anchor points
normal = torch.randn(B, N, D, requires_grad=True) # obj normals, computed by compute_mesh_normals
threshold = 1.0 # Threshold for filtering the near points on the object


force_mask = torch.zeros(B, M, dtype=torch.bool) # The regions that are contact, computed by the forces
force_mask[:, 1:15] = 1 # Example force mask, 1:15 contact

k_attr = torch.ones(B, M, dtype=torch.float32, requires_grad=True) * 1.5  # k_attr, computed by the forces
k_repl = 0.1  # k_repl

# Get the distance mask for batched points and reference points
distance_mask, dist_square, dist_vec = find_points_vec_dist_mask(points, ref_points, threshold)
print("Distance mask size:", distance_mask.size()) # (B, N, M), mask that identify which object verts are near the anchor points
print("Distances size:", dist_square.size()) # (B, N, M), squared distance between object verts and anchor points
print("Distances vector size:", dist_vec.size()) # (B, N, M, D), vector, v_obj - v_anchor


### Loss repl
print((dist_vec * normal.unsqueeze(2))[distance_mask].size())

### loss attr
print(dist_square[distance_mask * force_mask.unsqueeze(1)].size())

# Example loss computation for demonstration
loss_repl = 0.5 * k_repl * torch.exp(dist_vec * normal.unsqueeze(2))[distance_mask].sum()
print("Loss repl:", loss_repl)

# loss_attr = 0.5 * (k_attr.unsqueeze(1) * dist_square)[distance_mask * force_mask].sum()

k_attr = torch.zeros(B, M, dtype=torch.float32)
k_attr[:, 1:15] = 1
k_attr.requires_grad = True
loss_attr = 0.5 * (k_attr.unsqueeze(1) * dist_square)[distance_mask].sum()
print("Loss attr:", loss_attr)

loss = loss_repl + loss_attr
# Perform backpropagation
loss.backward()

# Check gradients
print("Gradients of points:\n", points.grad)
print("Gradients of ref_points:\n", ref_points.grad)
print("Gradients of normal:\n", normal.grad)
