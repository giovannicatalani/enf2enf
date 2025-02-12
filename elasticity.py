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
from utils.dataset import *
from utils.train_utils import *
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



@hydra.main(version_base=None, config_path="./configs/", config_name="config_elasticity")
def train(cfg: DictConfig):
    """Unified training function that handles both encoders and regression."""
    # Create base experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(cfg.logging.log_dir, f"etoe_elasticity_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save unified config
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    
    # Initialize wandb
    wandb.init(
        project=cfg.proj_name,
        name=f"etoe_elasticity_{timestamp}",
        dir=experiment_dir,
        config=OmegaConf.to_container(cfg),
    )
    
    # Setup datasets
    datasets = setup_datasets(cfg)
    

    # Create dataloaders
    train_loader_in = JAXDataLoader(
        datasets['in'][0], 
        batch_size=1, 
        shuffle=False
    )
    val_loader_in = JAXDataLoader(
        datasets['in'][1],
        batch_size= 1, 
        shuffle=False
    )
    
    # Get sample coordinates for model initialization
    sample_batch = next(iter(train_loader_in))
    coords = sample_batch['input']
    
    # Train geometry encoder
    cfg_in= cfg.encoder_in
    cfg_in.experiment_dir = os.path.join(experiment_dir, "encoder_in")
    geom_trainer, geom_state = train_encoder(
        cfg_in,
        'in',
        train_loader_in,
        val_loader_in,
        coords
    )
    
    print("Extracting geometry latents...")
    train_in_recon = geom_trainer.test(train_loader_in)
    val_in_recon = geom_trainer.test(val_loader_in)
    
    in_train_mse = np.mean((train_in_recon['predictions'] - train_in_recon['targets'])**2)
    in_val_mse = np.mean((val_in_recon['predictions'] - val_in_recon['targets'])**2)
    print(f"Train MSE geom: {in_train_mse:.6f}")
    print(f"Val MSE geom: {in_val_mse:.6f}")
    
    out_train_latents_ds = OutputLatentDataset(datasets['out'][0], train_in_recon,is_train=True,dataset_type='elasticity', use_conditions=False)
    train_stats = out_train_latents_ds.get_normalization_stats()
    print(train_stats['mean'], train_stats['std'])
    out_val_latents_ds   = OutputLatentDataset(datasets['out'][1], val_in_recon, is_train=False, train_stats=train_stats,dataset_type='elasticity', use_conditions=False)
    
    del train_loader_in, val_loader_in, datasets
    # 6) Wrap them in JAXDataLoader
    out_train_loader = JAXDataLoader(
        out_train_latents_ds, batch_size=4, shuffle=False
    )
    out_val_loader = JAXDataLoader(
        out_val_latents_ds, batch_size=1, shuffle=False
    )

    # 7) Train the output (pressure) ENF directly (no new latents)
    cfg_out = cfg.encoder_out
    cfg_out.experiment_dir = os.path.join(experiment_dir, "output_encoder")
    
    out_trainer, pressure_state = main_train_output_enf(
        cfg_out,
        out_train_loader,
        out_val_loader
    )

    wandb.finish()
    return 


if __name__ == "__main__":
    train()