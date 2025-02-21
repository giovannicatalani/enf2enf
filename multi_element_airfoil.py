import wandb
import numpy as np
import tqdm
import jax
import os
import jax.numpy as jnp 
from flax import struct
import optax
from functools import partial
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from omegaconf import OmegaConf
from enf.latents.autodecoder_meta import PositionOrientationFeatureAutodecoderMeta
from enf.models.equivariant_cross_attention_nef import EquivariantCrossAttentionNeF
from enf.steerable_attention.invariant.rel_pos import RelativePositionND
from utils.train_utils import *
from utils.dataset import *
from utils.trainer import train_encoder, main_train_output_enf, DirectOutputENFTrainer
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import tqdm
import flax.linen as nn
import yaml
import pickle
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.tri as tri


@hydra.main(version_base=None, config_path="./configs/", config_name="config_multielement_airfoil")
def train(cfg: DictConfig):
    # Create base experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(cfg.logging.log_dir, f"multiairfoil_training_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save unified config
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    
    # Initialize wandb
    wandb.init(
        project=cfg.proj_name,
        name=f"training_{timestamp}",
        dir=experiment_dir,
        config=OmegaConf.to_container(cfg),
    )
    
    # Setup datasets
    datasets = setup_datasets(cfg)
    

    geom_train_loader = JAXDataLoader(
        datasets['sdf'][0], 
        batch_size= 1, 
        shuffle=False
    )
    geom_val_loader = JAXDataLoader(
        datasets['sdf'][1],
        batch_size= 1, 
        shuffle=False
    )
    
    # Get sample coordinates for model initialization
    sample_batch = next(iter(geom_train_loader))
    coords = sample_batch['input']
    
    # Train geometry encoder
    geom_cfg = cfg.geometry_encoder
    geom_cfg.experiment_dir = os.path.join(experiment_dir, "sdf_encoder")
    geom_trainer, geom_state = train_encoder(
        geom_cfg,
        'sdf',
        geom_train_loader,
        geom_val_loader,
        coords
    )
    
    print("Extracting geometry latents...")
    train_geom_recon = geom_trainer.test(geom_train_loader)
    val_geom_recon = geom_trainer.test(geom_val_loader)
    
    geom_train_mse = np.mean((train_geom_recon['predictions'] - train_geom_recon['targets'])**2)
    geom_val_mse = np.mean((val_geom_recon['predictions'] - val_geom_recon['targets'])**2)
    print(f"Train MSE geom: {geom_train_mse:.6f}")
    print(f"Val MSE geom: {geom_val_mse:.6f}")
    
    
    # Save the validation reconstruction dictionary to disk
    save_path = os.path.join(experiment_dir, "val_geom_recon.npz")
    np.savez(save_path, **val_geom_recon)
    print(f"Saved val_geom_recon to {save_path}")
    wandb.finish()

    return 


if __name__ == "__main__":
    train()