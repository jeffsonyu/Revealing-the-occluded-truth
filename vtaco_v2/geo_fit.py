import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from networks.tracking_pipeline import *
from networks.geo_optimizer import GeOptimizer
from datasets.dataset import VTacODataModule

### Open3D summary for Mesh log
import open3d as o3d
# Verbose level: Error
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="config", config_name="fit_geo_default")
def main(cfg: DictConfig):
    # hydra creates working directory automatically
    print("Workding under: ", os.getcwd())
    os.makedirs("vis", exist_ok=True)
    
    if not os.path.exists(cfg.trainer.resume_from_checkpoint):
        raise FileNotFoundError("Checkpoint not found! Can't fit Geo without loaded model!")
    
    device = cfg.trainer.gpus[0]
    ### Problem with DISO!!!
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = "cuda:0"
    
    cfg_module_path = cfg.trackingmodule_path
    with open(cfg_module_path, "r") as f:
        cfg_module = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))

    datamodule = eval(cfg_module['datamodule_name'])(**cfg.datamodule)
    datamodule.setup()

    pipeline_model = eval(cfg_module['trackingmodule_name'])(**cfg_module.trackingmodule).to(device)
        
    # pipeline_model.load_model_only(cfg.trainer.resume_from_checkpoint)
    
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)

    logger = SummaryWriter("./default/{}".format(time.strftime("%Y-%m-%d-%H-%M")))
    
    geo_fitter = GeOptimizer(pipeline_model=pipeline_model,
                             **cfg.geo_param)
    
    for batch_idx, batch in enumerate(datamodule.val_dataloader()):
        name, *_ = batch.values()
        obj_mesh_o3d_list, hand_mesh_o3d_list = geo_fitter.fit_batch(batch, batch_idx)
        for i in range(len(name)):
            obj_mesh_o3d, hand_mesh_o3d = obj_mesh_o3d_list[i], hand_mesh_o3d_list[i]
            logger.add_3d(name[i], to_dict_batch([obj_mesh_o3d_list[i] + hand_mesh_o3d_list[i]]), step=batch_idx)
            
            o3d.io.write_triangle_mesh(f"./vis/{name[i]}.obj", obj_mesh_o3d)
            o3d.io.write_triangle_mesh(f"./vis/{name[i]}_hand.obj", hand_mesh_o3d)
        

# %%
# driver
if __name__ == "__main__":
    main()