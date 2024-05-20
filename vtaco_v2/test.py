import os
from icecream import ic
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import open3d as o3d

from networks.tracking_pipeline import *
from networks.callbacks import *
from datasets.dataset import *
from networks.transformer import *
from networks.hand_estimator import *
from networks.minipointnet import *

from networks.corr_module import *

from common.pointcloud_utils import *
from common.io_utils import *

from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

@hydra.main(config_path="config", config_name="train_tracking_default")
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.mkdir("checkpoints")

    datamodule = VTacODataModule(**cfg.datamodule)

    # pipeline_model = TestSDFTrackingModule(**cfg.trackingmodule)
    pipeline_model = WNFTrackingPipeline(**cfg.trackingmodule)
    
    ckpt = os.path.join(cfg.ckpt.ckpt_dir, "last.ckpt")
    
    logger = pl.loggers.TensorBoardLogger(
                save_dir=os.getcwd())

    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{iou:.4f}",
        monitor='iou',
        save_last=True,
        save_top_k=5,
        mode='max', 
        save_weights_only=False, 
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    vis_callback = VisCallBack()
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, vis_callback],
        checkpoint_callback=True,
        logger=logger,
        resume_from_checkpoint=ckpt, 
        **cfg.trainer)
    
    trainer.fit(model=pipeline_model, datamodule=datamodule)

    
if __name__ == "__main__":
    
    device = "cuda:7"
    x = torch.randn(8, 2048, 3).to(device)
    y = torch.randn(8, 2048, 3).to(device)
    
    with open("/mnt/homes/zhenjun/vtaco_v2/config/train_tracking_default.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open("/mnt/homes/zhenjun/vtaco_v2/config/train_tracking_corr.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    with open("/mnt/homes/zhenjun/vtaco_v2/config/train_tracking_corr_flow.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open("/mnt/homes/zhenjun/ViTaM/vtaco_v2/config/vitam.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        
    pc_1 = torch.rand(8, 1024, 3).to(device)
    pc_2 = torch.rand(8, 1024, 3).to(device)
    pred_1 = torch.rand(8, 4, 1024).to(device)
    
    pc_1_feat = torch.rand(8, 128, 1024).to(device)
    pc_2_feat = torch.rand(8, 128, 1024).to(device)
    # corr_fuser = CorrFlowFusion(**cfg['trackingmodule']['Corr_param']).to(device)
    # pred_corr_refined, pred_flow, feat_flow = corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
    # ic(pred_corr_refined.size(), pred_flow.size(), feat_flow.size())
    
    pred_flow = torch.rand(8, 1024, 3).to(device)
    pred_corr_refined = torch.rand(8, 1024, 1024).to(device)
    
    pc_2_corr = torch.einsum('bji,bik->bjk', pred_corr_refined, pc_2)
    ic(pc_2_corr.size())
    loss_flow = chamfer_distance(pc_1 + pred_flow, pc_2_corr)
    ic(loss_flow)

    loss_corr = 1 / 1024 - torch.mean(F.softmax(pred_corr_refined, dim=2))
    ic(loss_corr)
    
    
    # pc_1 = torch.rand(1, 1024, 3).to(device)
    # pointnet = PointNetfeat(**cfg['trackingmodule']['pointnet_param']).to(device)
    # pc_1_feat, pc_1_feat_global, *_ = pointnet(pc_1.permute(0, 2, 1))
    # ic(pc_1_feat.size(), pc_1_feat_global.size())
    
    dataset = VTacOTrackingForceDataset(**cfg['datamodule'])
    print(dataset.__len__())
    
    # module_name = "WNFTrackingCorrPipeline"
    # model = eval(cfg['trackingmodule_name'])(**cfg['trackingmodule']).to(device)
    # print(model.lr)
    
    # hand_model = HandTracker_v2(**cfg['trackingmodule']['hand_decoder_param']).to(device)
    # mano_out = hand_model(x, y)
    # ic(mano_out.size())
    
    
    # datamodule_pl = VTacODataModule(root_dir="/mnt/public/datasets/zhenjun/VTacO_Track/v00", 
    #                                 obj_class="001",
    #                                 tracking=True)
    # datamodule_pl.setup()
    # name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, force, *_ = datamodule_pl.dataset_train.__getitem__(0).values()
    # print(name, pc_1.shape, pc_2.shape, sample_p.shape, occ.shape, mano.shape, pc_for_norm.shape, force.shape)
    
    # unet = UNet3D(in_channels=3, out_channels=3).to(device)
    
    
    # pointnet = MiniPointNetfeat(nn_channels=(3, 16, 32, 48)).to(device)
    # # transformer = TransformerSiamese(fea_channels=(64, 32, 16, 1)).to(device)
    # transformer = TransformerFusion(fea_channels=(64, 32, 16, 1)).to(device)
    
    
    # # x_feat_unet = unet(x)
    # # ic(x_feat_unet.size())
    
    
    # x_feat, x_feat_global = pointnet(x.transpose(1, 2))
    # y_feat, y_feat_global = pointnet(y.transpose(1, 2))
    # ic(x_feat.size(), x_feat_global.size())
    # ic(y_feat.size(), y_feat_global.size())
    
    # # out = transformer.transform_fuse(x_feat, x, y_feat, y)
    # out = transformer(x_feat, x, y_feat, y)
    # ic(out.size())
    
    
    # pointnet = MiniPointNetfeat(**cfg['trackingmodule']['pointnet_param']).to(device)
    # hand_model = HandTracker(**cfg['trackingmodule']['hand_decoder_param']).to(device)
    
    # hand_pose_0 = 0.2 * torch.ones((1, 51), requires_grad=True).to(device)
    # out = hand_model.manolayer(hand_pose_0)
    # # ic(hand_model.manolayer.th_faces.cpu().numpy().shape, out[0][0].size())
    # # hand_mesh_0 = trimesh.Trimesh(out[0][0].detach().cpu().numpy(), hand_model.manolayer.th_faces.cpu().numpy())
    # # hand_mesh_0.export("flat_test.obj")
    
    # # out = hand_model(x.transpose(1, 2))
    
    
    # #[B,v,3]
    # tag = 'anchor'
    
    # mesh = trimesh.load_mesh('./assets/mano_hand_flat.obj')
    # mesh_faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)
    # #[456]
    # face_indices = torch.load(f"./assets/face_indices_{tag}.pt").to(device)
    # face_indices = face_indices.long()
    # #[456]
    # norm_off = torch.load(f"./assets/norm_off_{tag}.pt").float().requires_grad_().to(device)
    # #[456,3]
    # bary_coords = torch.load(f"./assets/bary_coords_{tag}.pt").float().requires_grad_().to(device)
    
    # verts = out[0]
    # num_poses = verts.shape[0]
    
    # #[b,456,3,3]
    # face_vertices = verts[:, mesh_faces[face_indices], :]
    # #v2-v1，v3-v2,[b,456,2,3]
    # edges = face_vertices[:, :, 1:, :] - face_vertices[:, :, :-1, :] 
    # #[b,456,3]
    # face_normals = torch.cross(edges[:, :, 0, :], edges[:, :, 1, :], dim=2)
    # face_normals = face_normals / face_normals.norm(dim=2, keepdim=True)
    # #[b,456,3]
    # bary_coords_expand = bary_coords.unsqueeze(0).expand(num_poses, -1, -1)
    # #[b,456,3]
    # projected_pos = torch.einsum('bijk,bij->bik', face_vertices, bary_coords_expand)

    # sensor_positions = projected_pos + norm_off.unsqueeze(0).unsqueeze(2) * face_normals
    # ic(projected_pos.size(), projected_pos.requires_grad)
    # # write_ply("sensor_positions.ply", sensor_positions[0].detach().cpu().numpy())

    # sensor_positions = projected_pos + projected_pos
    # y = sensor_positions.sum()
    # hand_pose_0.retain_grad()
    # y.backward()

    # ic(hand_pose_0.grad)

    
    
    # pc_1_feat, pc_1_feat_global = pointnet(x.permute(0, 2, 1))
    # ic(pc_1_feat.size())
    # ic(pc_1_feat_global.size())
    
    # x_test = torch.rand(8, 1024, 16).to(device)
    # pointnet_test = MiniPointNetfeat(nn_channels=(16, 32, 48, 64)).to(device)
    # x_test_feat, x_test_global = pointnet_test(x_test.permute(0, 2, 1))
    # ic(x_test_feat.size())
    
    
    # sample_p = torch.rand(8, 2048, 3).to(device)
    
    # pipeline_model = WNFTrackingCorrPipeline(**cfg['trackingmodule']).to(device)
    
    # out_wnf, mano_pred = pipeline_model(pc_1, pc_2, sample_p)
    # ic(out_wnf.size(), mano_pred.size())

    # main()