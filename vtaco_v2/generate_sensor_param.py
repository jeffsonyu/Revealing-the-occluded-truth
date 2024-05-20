import os
import numpy as np
import torch
import json
import trimesh
from common.io_utils import read_sensor_point_idx

    
def generate_sensor_pos_param(root_dir='.', mode='sensor'): 
    assert mode in ["sensor", "anchor"], "Mode unsupported!"

    sensor_positions, *_ = read_sensor_point_idx(os.path.join(root_dir, "sensor_idx.json"), mode=mode)
    mesh = trimesh.load_mesh(os.path.join(root_dir, "mano_hand_flat.obj"))


    nearest_vertices_indices = []   #最近的三个点
    nearest_face_indices = []       #面的索引
    face_normals = []               #法向量
    normal_offsets = []             #法向偏移
    projection_points = []          #投影点
    barycentric_coords = []         #重心坐标
    all_face_vertices = []          #每个面的三个点的顶点
    for i, sensor_pos in enumerate(sensor_positions):
        
        distances = np.linalg.norm(mesh.vertices - sensor_pos, axis=1)
        face_index = np.argmin(np.sum(distances[mesh.faces], axis=1))
        nearest_face_indices.append(face_index)

        vertex_index = mesh.faces[face_index]
        nearest_vertices_indices.append(vertex_index)  
        face_normal = np.cross(mesh.vertices[vertex_index[1]] - mesh.vertices[vertex_index[0]], mesh.vertices[vertex_index[2]] - mesh.vertices[vertex_index[0]])
        face_normal /= np.linalg.norm(face_normal)
        face_normals.append(face_normal)
        normal_offset = np.dot(face_normal, sensor_pos - mesh.vertices[vertex_index[0]])
        normal_offsets.append(normal_offset)      
        projected_pos = sensor_pos - normal_offset * face_normal
        projection_points.append(projected_pos)
        face_vertices = mesh.vertices[mesh.faces[face_index]]
        all_face_vertices.append(face_vertices)
        
    barycentric_coords = trimesh.triangles.points_to_barycentric(all_face_vertices, projection_points)
    bary_coords = torch.tensor(barycentric_coords, dtype=torch.float32)
    face_indices= torch.tensor(nearest_face_indices, dtype=torch.float32)
    norm_off = torch.tensor(normal_offsets, dtype=torch.float32)
    torch.save(bary_coords, os.path.join(root_dir, f'bary_coords_{mode}.pt'))
    torch.save(face_indices, os.path.join(root_dir, f'face_indices_{mode}.pt'))
    torch.save(norm_off, os.path.join(root_dir, f'norm_off_{mode}.pt'))

if __name__ == '__main__':
    root_dir = 'assets'
    generate_sensor_pos_param(root_dir=root_dir, mode='anchor')
    generate_sensor_pos_param(root_dir=root_dir, mode='sensor')