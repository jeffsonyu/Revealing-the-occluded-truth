import numpy as np
import torch
import open3d as o3d

def verts_faces_to_o3dmesh(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    ### normals are required for visualization, but can not be displayed in tensorboard, might cause warning
    mesh.compute_vertex_normals()
    return mesh


def compute_mesh_normals(vertices, faces):
    # vertices: FloatTensor, shape [num_vertices, 3]
    # faces: LongTensor, shape [num_faces, 3] - indices of vertices forming each face

    # Step 1: Calculate the normals for each face
    v0 = vertices[faces[:, 0]]  # First vertex of each face
    v1 = vertices[faces[:, 1]]  # Second vertex of each face
    v2 = vertices[faces[:, 2]]  # Third vertex of each face

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = torch.cross(edge1, edge2, dim=1)
    face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-6)  # Normalize

    # Step 2: Accumulate face normals to vertex normals
    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):  # Each face contributes to 3 vertices
        vertex_normals.index_add_(0, faces[:, i], face_normals)

    # Normalize the vertex normals
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-6)

    return vertex_normals

def compute_mesh_normals_batch(vertices, faces):
    # vertices: FloatTensor, shape [B, num_vertices, 3]
    # faces: LongTensor, shape [B, num_faces, 3] - indices of vertices forming each face

    B, num_vertices, _ = vertices.shape
    num_faces = faces.shape[1]

    # Step 1: Calculate the normals for each face
    # We use a batched gather operation to select vertices
    v0 = torch.gather(vertices, 1, faces[:, :, 0].unsqueeze(-1).expand(-1, -1, 3))
    v1 = torch.gather(vertices, 1, faces[:, :, 1].unsqueeze(-1).expand(-1, -1, 3))
    v2 = torch.gather(vertices, 1, faces[:, :, 2].unsqueeze(-1).expand(-1, -1, 3))

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = torch.cross(edge1, edge2, dim=2)
    face_normals = face_normals / (torch.norm(face_normals, p=2, dim=2, keepdim=True) + 1e-6)  # Normalize

    # Step 2: Accumulate face normals to vertex normals
    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):  # Each face contributes to 3 vertices
        vertex_normals.scatter_add_(1, faces[:, :, i].unsqueeze(-1).expand(-1, -1, 3), face_normals)

    # Normalize the vertex normals
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, p=2, dim=2, keepdim=True) + 1e-6)

    return vertex_normals