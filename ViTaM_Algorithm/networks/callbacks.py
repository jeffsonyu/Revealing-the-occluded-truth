import os
import numpy as np
import torch
import trimesh
import pytorch_lightning as pl

### Open3D tensorboard plugin
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from common.mesh_utils import verts_faces_to_o3dmesh
from common.io_utils import write_ply


class VisCallBack(pl.Callback):
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Verbose level: Error
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        
        current_epoch = trainer.fit_loop.current_epoch + 1
        global_step = trainer.fit_loop.global_step
        
        vis_every = pl_module.vis_every
    
        if current_epoch % vis_every == 0:
            item_name, occ_grid_list, verts_obj_list, faces_obj_list, mano_pred, verts_hand, faces_hand = pl_module.vis_step(batch, batch_idx)
            
            item_name = item_name[0]
            
            ### Use add_mesh to vis, Deprecated, use add_3d instead
            # verts_tensor, faces_tensor = torch.tensor(verts_obj.copy(), dtype=torch.float32).unsqueeze(0), torch.tensor(faces_obj.copy(), dtype=torch.int).unsqueeze(0)  
            # trainer.logger.experiment.add_mesh(tag=item_name, vertices=verts_tensor, faces=faces_tensor, global_step=global_step)
            
            mesh_obj_o3d = verts_faces_to_o3dmesh(verts_obj_list[0], faces_obj_list[0])
            mesh_obj_name = "{:03d}_{}.obj".format(current_epoch, item_name)
            
            o3d.io.write_triangle_mesh(os.path.join(trainer.logger.save_dir, "vis", mesh_obj_name), mesh_obj_o3d)

            mesh_log = mesh_obj_o3d
            
            if verts_hand is not None:
                mesh_hand_o3d = verts_faces_to_o3dmesh(verts_hand[0], faces_hand)
                mesh_log += mesh_hand_o3d
                
                o3d.io.write_triangle_mesh(os.path.join(trainer.logger.save_dir, "vis", mesh_obj_name.replace(".obj", "_hand.obj")), mesh_hand_o3d)
                
            if hasattr(pl_module, "vis_flow"):
                feat_flow = pl_module.vis_corr_flow(batch, batch_idx)
                ply_path = os.path.join(trainer.logger.save_dir, "vis", mesh_obj_name.replace(".obj", "_flow.ply"))
                write_ply(ply_path, feat_flow[0].transpose(0, 1).cpu().numpy())
                
                
            # if int(item_name.split("_")[1]) % 2 == 0:
            trainer.logger.experiment.add_3d(item_name, to_dict_batch([mesh_log]), step=global_step)