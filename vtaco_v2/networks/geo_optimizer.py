import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d
import trimesh
from diso import DiffDMC

from common.pointcloud_utils import chamfer_distance, norm_pointcloud_batch, find_points_vec_dist_mask, rotate_hand_points_tensor
from common.mesh_utils import verts_faces_to_o3dmesh, compute_mesh_normals, compute_mesh_normals_batch
from common.io_utils import read_sensor_point_idx, write_ply

class GeOptimizer():
    def __init__(self, pipeline_model, 
                       assets_dir,
                       sensor_sample_mode,
                       threshold_dist,
                       k_repl,
                       optimize_item,
                       loss_item,
                       max_steps,
                       lr=0.01):
        self.pipeline_model = pipeline_model
        
        ### GeO Parameters
        self.assets_dir = assets_dir
        self.sensor_sample_mode = sensor_sample_mode
        self.threshold_dist = threshold_dist
        self.k_repl = k_repl
        self.optimize_item = optimize_item
        self.loss_item = loss_item
        
        self.lr = lr
        self.max_steps = max_steps
        self.device = self.pipeline_model.device
        
        ### Load sensor-sample-required tensors
        sensor_idx_json = os.path.join(assets_dir, f"sensor_idx.json")
        _, self.region_name, self.region_idx = read_sensor_point_idx(sensor_idx_json, mode=sensor_sample_mode)
        
        hand_flat_mesh = trimesh.load_mesh(os.path.join(assets_dir, 'mano_hand_flat.obj'))
        self.hand_flat_faces = torch.tensor(hand_flat_mesh.faces, dtype=torch.int64).to(self.device)
        # faces indices
        self.hand_face_indices = torch.load(os.path.join(assets_dir, f"face_indices_{sensor_sample_mode}.pt")).long().to(self.device)
        # faces norm
        self.hand_norm_off = torch.load(os.path.join(assets_dir,f"norm_off_{sensor_sample_mode}.pt")).float().requires_grad_().to(self.device)
        # barycenter
        self.hand_bary_coords = torch.load(os.path.join(assets_dir, f"bary_coords_{sensor_sample_mode}.pt")).float().requires_grad_().to(self.device)
        
        
    
    def fit_batch(self, batch, batch_idx):
        ### Param to update: vertices of obj, mano poses
        # Have to put them on device...
        for key, value in batch.items():
            try:
                batch[key] = value.float().to(self.device)
            except:
                pass
            
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, sensor_forces, p_obj_gt, *_ = batch.values()
        B, N, C = pc_1.size()
        
        name, occ_grid_list, verts_obj_list, face_obj_list, mano_pred, verts_hand, face_hand = self.pipeline_model.vis_step(batch, batch_idx)

        verts_obj = torch.tensor([verts_obj_list[i].copy() for i in range(len(verts_obj_list))], dtype=torch.float32)
        occ_pred = torch.tensor([x.detach().cpu().numpy() for x in occ_grid_list], dtype=torch.float32)
        
        ### Tensor grad required
        occ_pred = occ_pred.to(self.device).requires_grad_()
        verts_obj = verts_obj.to(self.device).requires_grad_()
        # hand_pose = mano_pred.float().to(self.device).requires_grad_()
        hand_pose = mano.float().to(self.device).requires_grad_()
        
        hand_pose_init = hand_pose
        mano_verts_init, mano_faces_init = self.pipeline_model.forward_manolayer(hand_pose_init, pc_for_norm)
        verts_obj_init = verts_obj
        occ_pred_init = occ_pred

        param_optim = []
        if "hand" in self.optimize_item:
            param_optim.append({"params": [hand_pose]})
        if "sdf" in self.optimize_item:
            param_optim.append({"params": [occ_pred]})
    
        
        self.optimizer = torch.optim.Adam(param_optim, lr=self.lr)
        for e in range(self.max_steps):

            ### Define geo loss
            loss = 0
            sensor_positions = self.generate_sensor_position(hand_pose, pc_for_norm)
            
            extractors = [DiffDMC(dtype=torch.float32).to(self.device) for i in range(B)]
            for bs in range(B):
                verts_diso, faces_diso = self.pipeline_model.extract_mesh(extractors[bs], occ_pred[bs])
                if "Energy" in self.loss_item.keys():
                    ### Energy term
                    E_repl, E_attr = self.compute_energy(verts_diso.unsqueeze(0), faces_diso.unsqueeze(0), sensor_positions[bs:bs+1], sensor_forces)
                    print("E_repl:", E_repl.item())
                    print("E_attr:", E_attr.item())
                    loss += (self.loss_item["Energy"][0] * E_repl + self.loss_item["Energy"][1] * E_attr)
            
            if "hand_off" in self.loss_item.keys():
                ### Hand offset loss
                loss_hand_offset = self.loss_item["hand_off"] * F.mse_loss(hand_pose, hand_pose_init)
                loss += loss_hand_offset
            
            if "obj_off" in self.loss_item.keys():
                ### Obj verts offset loss
                loss_obj_offset = self.loss_item["obj_off"] * chamfer_distance(verts_diso.unsqueeze(0), verts_obj_init)
                loss += loss_obj_offset
            
            if "sdf_off" in self.loss_item.keys():
                loss_sdf_offset = self.loss_item["sdf_off"] * F.l1_loss(occ_pred.reshape(B, -1), occ_pred_init.reshape(B, -1))
                loss += loss_sdf_offset
            
            ### Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("Epoch {} with loss: {:.4f}".format(e, loss.item()))

        mesh_hand_o3d_list, mesh_obj_o3d_list = [], []
        mano_verts, mano_faces = self.pipeline_model.forward_manolayer(hand_pose, pc_for_norm)
        
        for i in range(mano_verts.size(0)):
            mesh_hand_o3d = verts_faces_to_o3dmesh(mano_verts[i].detach().cpu().numpy(), mano_faces.cpu().numpy())
            mesh_hand_o3d_list.append(mesh_hand_o3d)
        
        for i in range(occ_pred.size(0)):
            extractor = DiffDMC(dtype=torch.float32).to(self.device)
            verts_diso, faces_diso = self.pipeline_model.extract_mesh(extractor, occ_pred[i])
            
            verts_obj_i = verts_diso.detach().cpu().numpy()
            faces_obj_i = faces_diso.detach().cpu().numpy()
            mesh_obj_o3d = verts_faces_to_o3dmesh(verts_obj_i, faces_obj_i)
            mesh_obj_o3d_list.append(mesh_obj_o3d)
                
        return mesh_obj_o3d_list, mesh_hand_o3d_list
    
    
    def generate_sensor_position(self, mano_pose, pc_for_norm):
        mano_hand_trans, mano_hand_rot = mano_pose[:, :3], mano_pose[:, 3:]
        ### Unnormed mano vertices
        mano_verts, *_ = self.pipeline_model.hand_tracker.manolayer(mano_hand_rot)
        B, N, C = mano_verts.size()
        
        # [b,456,3,3]
        face_vertices = mano_verts[:, self.hand_flat_faces[self.hand_face_indices], :].float()
        # v2-v1，v3-v2,[b,456,2,3]
        edges = face_vertices[:, :, 1:, :] - face_vertices[:, :, :-1, :] 
        # [b,456,3]
        face_normals = torch.cross(edges[:, :, 0, :], edges[:, :, 1, :], dim=2)
        face_normals = face_normals / face_normals.norm(dim=2, keepdim=True)
        # [b,456,3]
        bary_coords_expand = self.hand_bary_coords.unsqueeze(0).expand(B, -1, -1)
        # [b,456,3]
        projected_pos = torch.einsum('bijk,bij->bik', face_vertices, bary_coords_expand)

        sensor_positions = projected_pos + self.hand_norm_off.unsqueeze(0).unsqueeze(2) * face_normals
        
        ### Transform and normalize the sensor points
        # sensor_positions += mano_hand_trans
        sensor_positions = rotate_hand_points_tensor(sensor_positions, mano_hand_trans, self.device)
        sensor_positions = norm_pointcloud_batch(sensor_positions, pc_for_norm)
    
        return sensor_positions
    
    
    def compute_energy(self, verts, faces, sensor_positions, sensor_forces):
        ### Compute Normals for the object
        normals = compute_mesh_normals_batch(verts, faces)
        
        ### Gather the forces for each region
        point_force_region = {}
        for r_name_i, r_idx_i in zip(self.region_name, self.region_idx):
            point_force_region[r_name_i] = sensor_forces[:, torch.tensor(r_idx_i)[:, 0], torch.tensor(r_idx_i)[:, 1]]

        ### K attractive string, probably computed primitively or with MLP
        if self.sensor_sample_mode == "anchor":
            k_attr = torch.stack([x.sum(dim=1) for x in point_force_region.values()], dim=1).to(self.device)
        elif self.sensor_sample_mode == "sensor":
            k_attr = torch.cat([x for x in point_force_region.values()], dim=1).to(self.device)
        
        ### Find the points that are within the threshold distance
        distance_mask, dist_square, dist_vec = find_points_vec_dist_mask(verts, sensor_positions, self.threshold_dist)
        
        ### Compute the energy
        E_repl = 0.5 * self.k_repl * torch.exp(dist_vec * normals.unsqueeze(2))[distance_mask].sum()
        E_attr = 0.5 * (k_attr.unsqueeze(1) * dist_square)[distance_mask].sum()
        
        return E_repl, E_attr