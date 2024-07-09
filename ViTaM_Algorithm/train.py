import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from networks.tracking_pipeline_force import ForceTrackingPipeline
from networks.callbacks import VisCallBack
from datasets.dataset import ViTaMDataModule


@hydra.main(config_path="config", config_name="test")
def main(cfg: DictConfig):
    # hydra creates working directory automatically
    print("Workding under: ", os.getcwd())
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("vis", exist_ok=True)

    if not os.path.exists(cfg.trainer.resume_from_checkpoint):
        cfg.trainer.resume_from_checkpoint = None
    
    datamodule = eval(cfg['datamodule_name'])(**cfg.datamodule)

    pipeline_model = eval(cfg['trackingmodule_name'])(**cfg.trackingmodule)
    
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
        **cfg.trainer)
    
    trainer.fit(model=pipeline_model, datamodule=datamodule)

# %%
# driver
if __name__ == "__main__":
    main()