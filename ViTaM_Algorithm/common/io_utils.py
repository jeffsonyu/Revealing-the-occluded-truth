import os
import numpy as np
import torch
import json
import trimesh
from plyfile import PlyData, PlyElement

### Deprecated
def read_sensor_point(json_file, mode='anchor'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    region_points = list(data.values())

    if mode == 'anchor':
        sensor_positions = [np.mean(x, axis=0) for x in region_points]
    
    if mode == 'sensor':
        sensor_positions = np.concatenate(region_points)
    
    return np.array(sensor_positions, dtype=np.float32)

def read_sensor_point_idx(json_file, mode='anchor'):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    region_name = list(data.keys())
    region_idx = [[eval(x) for x in list(region.keys())] for region in data.values()]
    region_points = [list(region.values()) for region in data.values()]

    if mode == 'anchor':
        sensor_positions = np.array([np.mean(x, axis=0) for x in region_points], dtype=np.float32)
    
    if mode == 'sensor':
        sensor_positions = np.concatenate(region_points, dtype=np.float32)
    
    return sensor_positions, region_name, region_idx

def write_ply(save_path, points, text=True):
    """
    save_path: path to save: '/yy/XX.ply'
    points: point_cloud: array or cpu tensor, size (N,3) or (N,6) where N is number of points
            (N,3) for x, y, z coordinates
            (N,6) for x, y, z coordinates and r, g, b colors
    text: if True, save as ASCII text format, else save as binary
    """
    if points.shape[1] == 6:  # Points with color
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elif points.shape[1] == 3:  # Points without color
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    else:
        raise ValueError("Points array must have shape (N,3) or (N,6)")

    # Convert tensor to numpy if necessary
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()

    # Create a structured array
    structured_array = np.array(list(map(tuple, points)), dtype=dtype)

    # Describe the vertex elements
    vertex_element = PlyElement.describe(structured_array, 'vertex', comments=['vertices'])

    # Write to a .PLY file
    PlyData([vertex_element], text=text).write(save_path)