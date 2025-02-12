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


class AirfransTrainer(DirectOutputENFTrainer):
    """ENF trainer with additional airfoil-specific results."""
    def extend_evaluation_results(self, results, all_targets):
        """Add airfoil-specific fields to results."""
        all_normals = []
        all_airfoil = []
        all_latents = []
        
        for batch in self.val_loader:
            all_latents.append(np.array(batch['latents_c']))
            all_normals.append(np.array(batch['normals']))
            all_airfoil.append(np.array(batch['is_airfoil']))
        
        results.update({
            'latent_features': np.concatenate(all_latents),
            'normals': np.concatenate(all_normals),
            'is_airfoil': np.concatenate(all_airfoil)
        })  


@hydra.main(version_base=None, config_path="./configs/", config_name="config_airfrans")
def train(cfg: DictConfig):
    """Unified training function that handles both encoders and regression."""
    # Create base experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(cfg.logging.log_dir, f"etoe_training_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save unified config
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    
    # Initialize wandb
    wandb.init(
        project=cfg.proj_name,
        name=f"etoe_training_{timestamp}",
        dir=experiment_dir,
        config=OmegaConf.to_container(cfg),
    )
    
    # Setup datasets
    datasets = setup_datasets(cfg)
    

    geom_train_loader = JAXDataLoader(
        datasets['geometry'][0], 
        batch_size= 1, 
        shuffle=False
    )
    geom_val_loader = JAXDataLoader(
        datasets['geometry'][1],
        batch_size= 1, 
        shuffle=False
    )
    
    # Get sample coordinates for model initialization
    sample_batch = next(iter(geom_train_loader))
    coords = sample_batch['input']
    
    # Train geometry encoder
    geom_cfg = cfg.geometry_encoder
    geom_cfg.experiment_dir = os.path.join(experiment_dir, "geometry_encoder")
    geom_trainer, geom_state = train_encoder(
        geom_cfg,
        'geometry',
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
    
    pressure_train_latents_ds = OutputLatentDataset(datasets['pressure'][0], train_geom_recon,is_train=True,dataset_type='airfrans', use_conditions=True)
    train_stats = pressure_train_latents_ds.get_normalization_stats()
    print(train_stats['mean'], train_stats['std'])
    pressure_val_latents_ds   = OutputLatentDataset(datasets['pressure'][1], val_geom_recon, is_train=False, train_stats=train_stats, dataset_type='airfrans', use_conditions=True)
    
    del geom_train_loader, geom_val_loader, datasets, train_geom_recon, val_geom_recon
    # 6) Wrap them in JAXDataLoader
    pressure_train_loader = JAXDataLoader(
        pressure_train_latents_ds, batch_size=4, shuffle=False
    )
    pressure_val_loader = JAXDataLoader(
        pressure_val_latents_ds, batch_size=1, shuffle=False
    )

    # 7) Train the output (pressure) ENF directly (no new latents)
    pressure_cfg = cfg.pressure_encoder
    pressure_cfg.experiment_dir = os.path.join(experiment_dir, "output_encoder")
    
    pressure_trainer, pressure_state = main_train_output_enf(
        pressure_cfg,
        pressure_train_loader,
        pressure_val_loader
    )

    wandb.finish()
    return 


if __name__ == "__main__":
    train()