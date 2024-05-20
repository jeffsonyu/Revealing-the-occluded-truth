import os
import time
import numpy as np
from skimage import measure
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl

from diso import DiffDMC

from networks.minipointnet import MiniPointNetfeat
from networks.pointnet import PointNetfeat
from networks.transformer import TransformerFusion
from networks.corr_module import CorrFusion, CorrFlowFusion
from components.unet3d import UNet3D
from networks.decoder import WNFdecoder
from networks.hand_estimator import HandTracker, HandTracker_v2, MLPDecoder

from common.eval_utils import compute_iou
from common.coordinate_utils import normalize_3d_coordinate, coordinate2index
from common.grid_utils import make_3d_grid
from common.pointcloud_utils import norm_pointcloud, norm_pointcloud_batch, rotate_hand_points_tensor, chamfer_distance

from torch_scatter import scatter_mean, scatter_max
### Load the checkpoint with this function
from pytorch_lightning.utilities.cloud_io import load as pl_load

torch.set_default_dtype(torch.float32)


class ForceTrackingPipeline(pl.LightningModule):
    def __init__(self,
                 training_param,
                 pointnet_param,
                 mlp_force_param,
                 transformer_force_param,
                 transformer_param,
                 grid_param,
                 unet3d_param,
                 decoder_param,
                 vis_param,
                 loss_items,
                 hand_decoder_param=None,
                 ):
        super().__init__()
        ### Training parameters
        self.lr = training_param['lr']
        self.optim = training_param['optimizer']
        
        assert self.optim in ["SGD", "Adam"], "Optimizer not supported"

        ### Pointnet
        self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        
        ### Force MLP
        self.force_mlp = MLPDecoder(**mlp_force_param)
        
        ### Transformer Force Fusion
        self.force_fuser = TransformerFusion(**transformer_force_param)
        
        
        ### Transformer Fusion
        self.transformerfuser = TransformerFusion(**transformer_param)
        
        ### Grid feature Generator
        self.grid_dim = grid_param['grid_dim']
        self.reso_grid = grid_param['reso_grid']
        self.padding = grid_param['padding']
        
        ### Grid feature Sampler
        self.sample_mode = grid_param['sample_mode']
        self.unet3d = UNet3D(**unet3d_param)
        

        ### WNF Decoder, inherit from ConvOccNet
        self.decoder = WNFdecoder(**grid_param, **decoder_param)
        
        ### Visualization param
        self.vis_every = vis_param['vis_every']
        self.nx = vis_param['nx']
        self.box_size = 1 + self.padding
        self.vis_point_every_split = vis_param['vis_point_every_split']
        self.print_vis_time = vis_param['print_time']
        
        ### Loss items
        self.loss_items = loss_items
    
    def load_model_only(self, chkpt_path):
        ckpt = pl_load(chkpt_path)
        self.load_state_dict(ckpt['state_dict'])
    
    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.grid_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x D x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.grid_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x D x reso^3)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid
    
    def sample_grid_feature(self, sample_p, c):
        p_nor = normalize_3d_coordinate(sample_p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        
        return c.permute(0, 2, 1)

    def forward_manolayer(self, hand_pose, pc_for_norm):
        mano_hand_trans, mano_hand_rot = hand_pose[:, :3], hand_pose[:, 3:]
        mano_info = self.hand_tracker.manolayer(mano_hand_rot)
        
        mano_verts, mano_jtrs, mano_transf, _ = mano_info
        # mano_verts += mano_hand_trans
        mano_verts = rotate_hand_points_tensor(mano_verts, mano_hand_trans, self.device)
        mano_verts = norm_pointcloud_batch(mano_verts, pc_for_norm)
        mano_faces = self.hand_tracker.manolayer.th_faces
        
        return mano_verts, mano_faces
    
    def forward_obj(self, pc_1, pc_2, force_1, force_2, anchors_1, anchors_2, sample_p):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Force MLP encode force features
        force_1_feat = self.force_mlp(force_1)
        force_2_feat = self.force_mlp(force_2)
        
        ### Transformer Fusion for force and pc feat
        force_feat_fused_1 = self.force_fuser(pc_1_feat, pc_1, force_1_feat, force_1)
        force_feat_fused_2 = self.force_fuser(pc_2_feat, pc_2, force_2_feat, force_2)
        
        ### Transformer Fusion feature from two frames
        pc_feat_fused = self.transformerfuser(force_feat_fused_2, pc_2, force_feat_fused_1, pc_1)
        
        ### Generate grid features from pc_2 and fused_feat
        pc_feat_fused_grid = self.generate_grid_features(pc_2, pc_feat_fused)
        
        ### Sample feature for sample points from grid_feat
        sample_p_feat = self.sample_grid_feature(sample_p, pc_feat_fused_grid)
        
        ### Decode wnf from sample_p_feat
        out_wnf = self.decoder(sample_p, sample_p_feat)
        
        return out_wnf
    
    def forward(self, pc_1, pc_2, force_1, force_2, mano_1, mano_2, sample_p):
        ### Calculate anchor points position
        anchors_1 = self.generate_anchors_mano(mano_1)
        anchors_2 = self.generate_anchors_mano(mano_2)
        
        ### Predict wnf for sample_p
        out_wnf = self.forward_obj(pc_1, pc_2, force_1, force_2, anchors_1, anchors_2, sample_p)
        
        ### Predict hand pose
        if self.hand_tracker is not None:
            mano_pred = self.hand_tracker(pc_1, pc_2)
            return out_wnf, mano_pred
        
        return out_wnf, None

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, forces, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        occ_pred, mano_pred = self.forward(pc_1, pc_2, sample_p)
        
        if "sdf" in self.loss_items.keys():
            loss = self.loss_items['sdf'] * F.l1_loss(occ_pred, occ)
            self.log("occ_loss", loss)
        
        if (mano_pred is not None) and ("hand" in self.loss_items.keys()):
            loss_hand = self.loss_items['hand'] * F.mse_loss(mano_pred, mano)
            self.log("hand_loss", loss_hand)
            
            loss += loss_hand
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ = pc_1.float(), pc_2.float(), sample_p.float(), occ.float()
        
        occ_pred, *_ = self.forward(pc_1, pc_2, sample_p)
        iou = compute_iou(occ1=occ.detach().cpu(), occ2=occ_pred.detach().cpu(), threshold=0.5)[0]
        print("Validation metric (iou):", iou)
        self.log("iou", iou)
        
        return iou
    
    def extract_mesh(self, extractor, occ_grid):
        verts_diso, faces_diso = extractor(occ_grid)
        verts_diso -= 0.5 * torch.ones(3).to(self.device)
        verts_diso *= (1 + self.padding)

        return verts_diso, faces_diso
    
    def vis_obj(self, pc_1, pc_2, sample_p_grid_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            occ_pred_list = []
            for sample_p in sample_p_grid_split:
                sample_p = sample_p.to(self.device)
                occ_pred = self.forward_obj(pc_1, pc_2, sample_p.unsqueeze(0))
                occ_pred_list.append(occ_pred)
        
        occ_pred = torch.cat(occ_pred_list, dim=1).to(self.device)
        
        occ_grid = occ_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, occ_grid)
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return occ_grid, vertices, faces
    
    def vis_hand(self, pc_1, pc_2, pc_for_norm):
        ### Used on callback for visulization, or geo fit
        mano_pred = self.hand_tracker(pc_1, pc_2)
        mano_verts, mano_faces = self.forward_manolayer(mano_pred, pc_for_norm)

        return mano_pred, mano_verts, mano_faces
    
    def vis_step(self, batch, batch_idx):
        ### Used on callback for visulization, or geo fit
        start_vis_time = time.time()
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, occ, mano, pc_for_norm = pc_1.float(), pc_2.float(), occ.float(), mano.float(), pc_for_norm.float()
        
        B, N, C = pc_2.size()
        sample_p_grid = self.box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (self.nx,)*3
            )
        sample_p_grid_split = torch.split(sample_p_grid, self.vis_point_every_split)
        
        with torch.no_grad():
            occ_grid_list, verts_obj_list, face_obj_list = [], [], []
            for b in range(B):
                pc_1_batch, pc_2_batch = pc_1[b:b+1], pc_2[b:b+1]
                occ_grid, verts_obj, face_obj = self.vis_obj(pc_1_batch, pc_2_batch, sample_p_grid_split)
                verts_obj, face_obj = np.array(verts_obj, dtype=np.float32), np.array(face_obj, dtype=np.int32)

                occ_grid_list.append(occ_grid)
                verts_obj_list.append(verts_obj)
                face_obj_list.append(face_obj)
            
            if self.print_vis_time:
                print("Vis time: ", time.time() - start_vis_time)

            if self.hand_tracker is not None:
                mano_pred, verts_hand, face_hand = self.vis_hand(pc_1_batch, pc_2_batch, pc_for_norm)
                
                ### Test if hand vis is correct
                # mano_pred = None
                verts_hand, face_hand = self.forward_manolayer(mano, pc_for_norm)
                
                verts_hand, face_hand = verts_hand.detach().cpu().numpy(), face_hand.cpu().numpy()
                
            else:
                verts_hand, face_hand = None, None
                    
        return name, occ_grid_list, verts_obj_list, face_obj_list, mano_pred, verts_hand, face_hand

    def configure_optimizers(self):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer