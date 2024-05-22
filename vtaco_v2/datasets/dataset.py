import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

import common.transform_utils as transform_utils


class VTacODataset(Dataset):
    def __init__(self, root_dir, obj_class, tracking=False, train_all=False, dataset_split='train', sample_point_num=2048, sample_pc_num=2048, pc_noise=0.005) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mesh_dir = os.path.join(self.root_dir, "VTacO_mesh")
        
        self.obj_class_list = ['YCB', 'bottle', 'box', 'cutter', 'foldingrack', 'lock', 'scissor', 'tableware']
        
        self.dataset_split = dataset_split
        
        self.transform_point = transform_utils.SubsamplePoints(sample_point_num)
        self.transform_pc = transforms.Compose([transform_utils.SubsamplePointcloud(sample_pc_num), 
                                                transform_utils.PointcloudNoise(pc_noise)])
        
        
        self.tracking = tracking
        
        self.name_list = []
        self.pc_list = []
        self.pc_for_norm_list = []
        
        self.sample_points_list = []
        self.occ_list = []
        self.mano_list = []
        self.points_obj_gt_list = []

        
        if not train_all:
            self.prepare_obj_class_data(obj_class)
        
        else:
            for obj_class_name in self.obj_class_list[1:]:
                obj_class = "{:03d}".format(self.obj_class_list.index(obj_class_name))
                self.prepare_obj_class_data(obj_class)
    
    def prepare_obj_class_data(self, obj_class):
        obj_class_dir = os.path.join(self.root_dir, "VTacO_AKB_class", obj_class, self.obj_class_list[int(obj_class)])
        
        with open(os.path.join(obj_class_dir, f"{self.dataset_split}.lst"), "r") as f:
            self.datapoint_list = [line.strip() for line in f.readlines()]
            
        for data_item in os.listdir(obj_class_dir):
            if data_item.endswith(".lst"):
                continue
            
            if data_item in self.datapoint_list:
                points = np.load(os.path.join(obj_class_dir, data_item, "points.npz"), allow_pickle=True)
            
                pointcloud = np.load(os.path.join(obj_class_dir, data_item, "pointcloud.npz"), allow_pickle=True)
                
                # print(dict(points).keys(), dict(pointcloud).keys())
                
                points = self.transform_point(dict(points))
                pointcloud = self.transform_pc(dict(pointcloud))

                
                self.name_list.append(data_item)
                self.pc_list.append(pointcloud['points'])
                self.pc_for_norm_list.append(pointcloud['pc_ply'])
                
                self.sample_points_list.append(points['points'])
                self.occ_list.append(points['occupancies'])
                self.mano_list.append(points['mano'].reshape(-1))
                self.points_obj_gt_list.append(points['points_obj'])
    
    
    def __len__(self):
        if self.tracking:
            return len(self.pc_list) - 1
        return len(self.pc_list)

    def __getitem__(self, index):
        item_dict = dict()
        
        if self.tracking:
            item_dict['name'] = self.name_list[index+1]
            item_dict['pc_1'] = self.pc_list[index]
            item_dict['pc_2'] = self.pc_list[index+1]
            item_dict['sample_points'] = self.sample_points_list[index+1]
            item_dict['occ'] = self.occ_list[index+1]
            item_dict['mano'] = self.mano_list[index+1]
            item_dict['pc_for_norm'] = self.pc_for_norm_list[index+1]
            item_dict['points_obj_gt'] = self.points_obj_gt_list[index+1]
        
        else:
            item_dict['name'] = self.name_list[index+1]
            item_dict['pc'] = self.pc_list[index]
            item_dict['sample_points'] = self.sample_points_list[index]
            item_dict['occ'] = self.occ_list[index]
            item_dict['mano'] = self.mano_list[index]
            item_dict['pc_for_norm'] = self.pc_for_norm_list[index]
            item_dict['points_obj_gt'] = self.points_obj_gt_list[index]
        
        return item_dict


class VTacOTrackingDataset(Dataset):
    def __init__(self, root_dir, obj_class, tracking=False, train_all=False, dataset_split='train', sample_point_num=2048, sample_pc_num=2048, pc_noise=0.005) -> None:
        super().__init__()
        self.root_dir = root_dir
        # self.mesh_dir = os.path.join(self.root_dir, "VTacO_mesh")
        
        self.obj_class_list = ['YCB', 'bottle', 'box', 'cutter', 'foldingrack', 'lock', 'scissor', 'tableware']
        
        self.dataset_split = dataset_split
        
        self.transform_point = transform_utils.SubsamplePoints(sample_point_num)
        self.transform_pc = transforms.Compose([transform_utils.SubsamplePointcloud(sample_pc_num), 
                                                transform_utils.PointcloudNoise(pc_noise)])
        
        
        self.tracking = tracking
        
        self.name_list = []
        self.pc_list = []
        self.pc_for_norm_list = []
        
        self.sample_points_list = []
        self.occ_list = []
        self.mano_list = []
        self.points_obj_gt_list = []
        
        self.point_force = []

        
        if not train_all:
            self.prepare_obj_class_data(obj_class)
        
        else:
            for obj_class_name in self.obj_class_list[1:]:
                obj_class = "{:03d}".format(self.obj_class_list.index(obj_class_name))
                self.prepare_obj_class_data(obj_class)
    
    def prepare_obj_class_data(self, obj_class):
        obj_class_dir = os.path.join(self.root_dir, obj_class)
        
        with open(os.path.join(obj_class_dir, f"{self.dataset_split}.lst"), "r") as f:
            self.datapoint_list = [line.strip() for line in f.readlines()]
            seq_depth_dir_list = []
            for datapoint in self.datapoint_list:
                datapoint_dir = obj_class_dir
                for data_dir in datapoint.split("_"):
                    datapoint_dir = os.path.join(datapoint_dir, data_dir)
                seq_depth_dir_list.append(datapoint_dir)
        
        # for seq_depth_dir in seq_depth_dir_list:
        for datapoint_name in self.datapoint_list:
            seq_depth_dir = obj_class_dir
            for data_dir in datapoint_name.split("_"):
                seq_depth_dir = os.path.join(seq_depth_dir, data_dir)
                
            if not os.path.exists(seq_depth_dir): continue
            for frame_name in os.listdir(seq_depth_dir):
                frame_dir = os.path.join(seq_depth_dir, frame_name)
                points = np.load(os.path.join(frame_dir, "points.npz"), allow_pickle=True)
            
                pointcloud = np.load(os.path.join(frame_dir, "pointcloud.npz"), allow_pickle=True)
                
                points = self.transform_point(dict(points))
                pointcloud = self.transform_pc(dict(pointcloud))
                if self.tracking:
                    if frame_name.endswith("001"):
                        pc_former = pointcloud['points']
                    else:
                        pc_now = pointcloud['points']
                        self.name_list.append(datapoint_name + "_" + frame_name)
                        self.pc_list.append([pc_former, pc_now])
                        self.pc_for_norm_list.append(pointcloud['pc_ply'])
                        self.sample_points_list.append(points['points'])
                        self.occ_list.append(points['occupancies'])
                        self.mano_list.append(points['mano'].reshape(-1))
                        self.points_obj_gt_list.append(points['points_obj'])
                        self.point_force.append(pointcloud['force'])
                        pc_former = pc_now
                else:
                    self.name_list.append(datapoint_name + "_" + frame_name)
                    self.pc_list.append(pointcloud['points'])
                    self.pc_for_norm_list.append(pointcloud['pc_ply'])
                    self.sample_points_list.append(points['points'])
                    self.occ_list.append(points['occupancies'])
                    self.mano_list.append(points['mano'].reshape(-1))
                    self.points_obj_gt_list.append(points['points_obj'])
                    self.point_force.append(pointcloud['force'])
    
    
    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, index):
        item_dict = dict()

        if self.tracking:
            item_dict['name'] = self.name_list[index]
            item_dict['pc_1'] = self.pc_list[index][0]
            item_dict['pc_2'] = self.pc_list[index][1]
            item_dict['sample_points'] = self.sample_points_list[index]
            item_dict['occ'] = self.occ_list[index]
            item_dict['mano'] = self.mano_list[index]
            item_dict['pc_for_norm'] = self.pc_for_norm_list[index]
            item_dict['force'] = self.point_force[index]
            item_dict['points_obj_gt'] = self.points_obj_gt_list[index]
        else:
            item_dict['name'] = self.name_list[index]
            item_dict['pc'] = self.pc_list[index]
            item_dict['sample_points'] = self.sample_points_list[index]
            item_dict['occ'] = self.occ_list[index]
            item_dict['mano'] = self.mano_list[index]
            item_dict['pc_for_norm'] = self.pc_for_norm_list[index]
            item_dict['force'] = self.point_force[index]
            item_dict['points_obj_gt'] = self.points_obj_gt_list[index]
        
        return item_dict


class VTacOTrackingForceDataset(Dataset):
    def __init__(self, root_dir, obj_class, tracking=False, train_all=False, dataset_split='train', sample_point_num=2048, sample_pc_num=2048, pc_noise=0.005) -> None:
        super().__init__()
        self.root_dir = root_dir
        # self.mesh_dir = os.path.join(self.root_dir, "VTacO_mesh")
        
        self.obj_class_list = ['YCB', 'bottle', 'box', 'cutter', 'foldingrack', 'lock', 'scissor', 'tableware']
        
        self.dataset_split = dataset_split
        
        self.transform_point = transform_utils.SubsamplePoints(sample_point_num)
        self.transform_pc = transforms.Compose([transform_utils.SubsamplePointcloud(sample_pc_num), 
                                                transform_utils.PointcloudNoise(pc_noise)])
        
        
        self.tracking = tracking
        
        self.name_list = []
        self.pc_list = []
        self.pc_for_norm_list = []
        
        self.sample_points_list = []
        self.occ_list = []
        self.mano_list = []
        self.points_obj_gt_list = []
        
        self.point_force = []

        
        if not train_all:
            self.prepare_obj_class_data(obj_class)
        
        else:
            for obj_class_name in self.obj_class_list[1:]:
                obj_class = "{:03d}".format(self.obj_class_list.index(obj_class_name))
                self.prepare_obj_class_data(obj_class)
    
    def prepare_obj_class_data(self, obj_class):
        obj_class_dir = os.path.join(self.root_dir, obj_class)
        
        with open(os.path.join(obj_class_dir, f"{self.dataset_split}.lst"), "r") as f:
            self.datapoint_list = [line.strip() for line in f.readlines()]
            seq_depth_dir_list = []
            for datapoint in self.datapoint_list:
                datapoint_dir = obj_class_dir
                for data_dir in datapoint.split("_"):
                    datapoint_dir = os.path.join(datapoint_dir, data_dir)
                seq_depth_dir_list.append(datapoint_dir)
        
        # for seq_depth_dir in seq_depth_dir_list:
        for datapoint_name in self.datapoint_list:
            seq_depth_dir = obj_class_dir
            for data_dir in datapoint_name.split("_"):
                seq_depth_dir = os.path.join(seq_depth_dir, data_dir)
                
            if not os.path.exists(seq_depth_dir): continue
            for frame_name in os.listdir(seq_depth_dir):
                frame_dir = os.path.join(seq_depth_dir, frame_name)
                points = np.load(os.path.join(frame_dir, "points.npz"), allow_pickle=True)
                pointcloud = np.load(os.path.join(frame_dir, "pointcloud.npz"), allow_pickle=True)
                
                points = self.transform_point(dict(points))
                pointcloud = self.transform_pc(dict(pointcloud))
                if self.tracking:
                    if frame_name.endswith("001"):
                        pc_former = pointcloud['points']
                        mano_former = points['mano'].reshape(-1)
                        force_former = pointcloud['force']
                    
                    elif "Frame{:03d}".format(int(frame_name[-3:])+1) not in os.listdir(seq_depth_dir):
                        continue
                    
                    else:
                        ### Next Frame data
                        next_frame = "Frame{:03d}".format(int(frame_name[-3:])+1)
                        next_frame_dir = os.path.join(seq_depth_dir, next_frame)
                        points_next = np.load(os.path.join(next_frame_dir, "points.npz"), allow_pickle=True)
                        pointcloud_next = np.load(os.path.join(next_frame_dir, "pointcloud.npz"), allow_pickle=True)
                        points_next = self.transform_point(dict(points_next))
                        pointcloud_next = self.transform_pc(dict(pointcloud_next))
                        
                        pc_next = pointcloud_next['points']
                        force_next = pointcloud_next['force']
                        mano_next = points_next['mano'].reshape(-1)
                        
                        ### This frame data
                        pc_now = pointcloud['points']
                        force_now = pointcloud['force']
                        mano_now = points['mano'].reshape(-1)
                        
                        self.name_list.append(datapoint_name + "_" + frame_name)
                        self.pc_list.append(np.array([pc_former, pc_now, pc_next]))
                        self.pc_for_norm_list.append(pointcloud['pc_ply'])
                        self.sample_points_list.append(points['points'])
                        self.occ_list.append(points['occupancies'])
                        self.mano_list.append(np.array([mano_former, mano_now, mano_next]))
                        self.points_obj_gt_list.append(points['points_obj'])
                        self.point_force.append(np.array([force_former, force_now, force_next]))
                        
                        pc_former = pc_now
                        force_former = force_now
                        mano_former = mano_now
                else:
                    self.name_list.append(datapoint_name + "_" + frame_name)
                    self.pc_list.append(pointcloud['points'])
                    self.pc_for_norm_list.append(pointcloud['pc_ply'])
                    self.sample_points_list.append(points['points'])
                    self.occ_list.append(points['occupancies'])
                    self.mano_list.append(points['mano'].reshape(-1))
                    self.points_obj_gt_list.append(points['points_obj'])
                    self.point_force.append(pointcloud['force'])
    
    
    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, index):
        item_dict = dict()

        if self.tracking:
            item_dict['name'] = self.name_list[index]
            item_dict['pc'] = self.pc_list[index]
            item_dict['sample_points'] = self.sample_points_list[index]
            item_dict['occ'] = self.occ_list[index]
            item_dict['mano'] = self.mano_list[index]
            item_dict['pc_for_norm'] = self.pc_for_norm_list[index]
            item_dict['force'] = self.point_force[index]
            item_dict['points_obj_gt'] = self.points_obj_gt_list[index]
        else:
            item_dict['name'] = self.name_list[index]
            item_dict['pc'] = self.pc_list[index]
            item_dict['sample_points'] = self.sample_points_list[index]
            item_dict['occ'] = self.occ_list[index]
            item_dict['mano'] = self.mano_list[index]
            item_dict['pc_for_norm'] = self.pc_for_norm_list[index]
            item_dict['force'] = self.point_force[index]
            item_dict['points_obj_gt'] = self.points_obj_gt_list[index]
        
        return item_dict
    

class VTacODataModule(pl.LightningDataModule):
    def __init__(self, root_dir, obj_class, 
                 tracking=False, train_all=False,  
                 sample_point_num=2048, 
                 sample_pc_num=2048, 
                 pc_noise=0.005, 
                 batch_size=8,
                 num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.root_dir = root_dir
        self.obj_class = obj_class
        self.tracking = tracking
        self.train_all = train_all
        self.sample_point_num = sample_point_num
        self.sample_pc_num = sample_pc_num
        self.pc_noise = pc_noise
        
        # self.dataset = VTacOTrackingDataset if "Track" in self.root_dir else VTacODataset
        self.dataset = VTacOTrackingForceDataset

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # split dataset
        if stage in (None, "fit"):
            self.dataset_train = self.dataset(root_dir=self.root_dir,
                                              obj_class=self.obj_class,
                                              train_all=self.train_all,
                                              tracking=self.tracking,
                                              dataset_split='train',
                                              sample_point_num=self.sample_point_num,
                                              sample_pc_num=self.sample_pc_num,
                                              pc_noise=self.pc_noise)
            
            self.dataset_val = self.dataset(root_dir=self.root_dir,
                                            obj_class=self.obj_class,
                                            train_all=self.train_all,
                                            tracking=self.tracking,
                                            dataset_split='val',
                                            sample_point_num=self.sample_point_num,
                                            sample_pc_num=self.sample_pc_num,
                                            pc_noise=self.pc_noise)
            
        if stage in (None, "test"):
            self.dataset_test = self.dataset(root_dir=self.root_dir,
                                             obj_class=self.obj_class,
                                             train_all=self.train_all,
                                             tracking=self.tracking,
                                             dataset_split='test',
                                             sample_point_num=self.sample_point_num,
                                             sample_pc_num=self.sample_pc_num,
                                             pc_noise=self.pc_noise)

    # return the dataloader for each split
    def train_dataloader(self):
        dataloader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return dataloader_train

    def val_dataloader(self):
        dataloader_val = DataLoader(self.dataset_val, batch_size=1, num_workers=self.num_workers)
        return dataloader_val

    def test_dataloader(self):
        dataloader_test = DataLoader(self.dataset_test, batch_size=1, num_workers=self.num_workers)
        return dataloader_test