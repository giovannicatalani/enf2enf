import numpy as np
import wandb
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
from omegaconf import DictConfig, OmegaConf
import datetime
import scienceplots

def init_nef(cfg):
    """Initialize NeF model from config."""
    return EquivariantCrossAttentionNeF(
        num_hidden=cfg.model.num_hidden,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        num_out=cfg.model.num_out,
        latent_dim=cfg.model.latent_dim,
        cross_attn_invariant=RelativePositionND(num_dims=cfg.model.coord_dim),
        self_attn_invariant=RelativePositionND(num_dims=cfg.model.coord_dim),
        embedding_type=cfg.model.embedding_type,
        embedding_freq_multiplier=cfg.model.embedding_freq_multiplier,
        condition_value_transform=cfg.model.condition_value_transform
    )  
    
def train_encoder(cfg, dataset_type, train_loader, val_loader, coords):
    """Train a single encoder."""
    print(f"\nTraining {dataset_type} encoder...")
    
    # Initialize NeF model
    nef = init_nef(cfg)
    
    # Create trainer
    trainer = MetaSGDSteadyStateTrainer(
        config=cfg,
        nef=nef,
        train_loader=train_loader,
        val_loader=val_loader,
        coords=coords,
        seed=cfg.seed
    )
    
    # Train model
    final_state = trainer.train_model(cfg.training.num_epochs)
    
    return trainer, final_state

def main_train_output_enf(cfg, train_loader, val_loader):
    """
    Instantiate and train the DirectOutputENFTrainer that:
      - Initializes p, w once via PositionOrientationFeatureAutodecoder
      - FREEZES them (not in param tree)
      - Uses geometry latents 'a' from each batch
      - Trains only the ENF weights
    """
    trainer = DirectOutputENFTrainer(cfg, train_loader, val_loader, seed=cfg.seed)
    final_state = trainer.train_model(cfg.training.num_epochs)
    return trainer, final_state


class DirectOutputENFTrainer:
    """
    Trainer that:
      - Initializes positions/orientations (p) + Gaussian window (w) once using
        PositionOrientationFeatureAutodecoderMeta with num_signals=1.
      - Uses geometry latents a from batch['latents_c'].
      - Does NOT update p, w (they're frozen).
     """
    
    @struct.dataclass
    class TrainState:
        """Container for ENF parameters and optimizer state."""
        params: dict         
        opt_state: optax.OptState
        rng: jnp.ndarray

    def __init__(self, cfg, train_loader, val_loader, seed=0):
        """
        Args:
          cfg: Config/dict (must have model + training fields).
          train_loader: yields batches with:
             - 'input': [B, N, coord_dim]  (PDE domain coords)
             - 'latents_c': [B, L, latent_dim(+cond)] (geom-latent features)
             - 'output': [B, N, out_dim]   (PDE solution)
          val_loader: same structure
          seed: random seed
        """
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed

        # 1) Initialize the ENF
        self.enf = EquivariantCrossAttentionNeF(
            num_hidden=cfg.model.num_hidden,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            num_out=cfg.model.num_out,
            latent_dim=cfg.model.latent_dim,
            cross_attn_invariant=RelativePositionND(num_dims=cfg.model.coord_dim),
            self_attn_invariant=RelativePositionND(num_dims=cfg.model.coord_dim),
            embedding_type=cfg.model.embedding_type,
            embedding_freq_multiplier=cfg.model.embedding_freq_multiplier,
            condition_value_transform=cfg.model.condition_value_transform,
            use_enhanced_stem=True
        )    # e.g. EquivariantCrossAttentionNeF

        # 2) Build a brand-new autodecoder for positions, window, etc.

        self.output_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=1,
            num_latents=cfg.model.num_latents,
            latent_dim=cfg.model.latent_dim,   # unused 'a', but needed for shape
            spatial_dims=cfg.model.spatial_dim,
            num_pos_dims=self.enf.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=self.enf.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.model.gaussian_window,
            coordinate_system="cartesian",
            bbox=cfg.model.latent_bbox
        )

        # 3) Random keys
        key = jax.random.PRNGKey(self.seed)
        key, init_nef_key, init_dec_key = jax.random.split(key, 3)

        # 4) Initialize the autodecoder once
        dec_params = self.output_autodecoder.init(init_dec_key)
        p_init, a_init, w_init = self.output_autodecoder.apply(dec_params)
        # p_init: [1, L, pos_dim(+ori_dim)]
        # w_init: [1, L, 1] or possibly None

        # Freeze them:
        self.frozen_p = p_init  # shape [1, L, ...]
        self.frozen_w = w_init  # shape [1, L, 1] or None

        # 5) Initialize ENF params for shape
        dummy_coords = jnp.zeros((1, 16, cfg.model.coord_dim))
        nef_params = self.enf.init(
            init_nef_key,
            dummy_coords,
            self.frozen_p,  # just for shape
            a_init,         # from autodecoder, also just for shape
            self.frozen_w
        )

        # We'll store only the ENF parameters in the param tree (p,w are frozen).
        all_params = {'nef': nef_params}
        
        self.scheduler = ReduceLROnPlateau(
            initial_lr=cfg.optimizer.learning_rate_enf,
            factor=cfg.optimizer.get('lr_factor', 0.95),
            patience=cfg.optimizer.get('lr_patience', 50),
            min_lr=cfg.optimizer.get('min_lr', 1e-6),
            min_delta=cfg.optimizer.get('min_delta', 1e-4)
        )

        # 6) Build an optimizer (only for self.enf)
        self.current_lr = self.scheduler.get_lr()
        self.optimizer = optax.adam(self.current_lr)
        opt_state = self.optimizer.init(all_params)

        # 7) Build initial TrainState
        self.state = self.TrainState(
            params=all_params,
            opt_state=opt_state,
            rng=key
        )

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch):
        """
        Single gradient update step with random coordinate mask.

        batch keys:
         - 'input': [B, N, coord_dim]
         - 'latents_c': [B, L, latent_dim(+cond)]
         - 'output': [B, N, out_dim]
        """
        def loss_fn(params, rng):
            coords_full = batch['input']      # [B, N, coord_dim]
            targets_full = batch['output']    # [B, N, out_dim]
            B, N, _ = coords_full.shape

            # (A) Generate a random mask for sub-sampling coords
            num_sampled = self.cfg.training.num_sampled_points
            # We do a single random permutation for each batch, for each device
            # If you want multi-step, adapt as needed
            mask = jax.random.permutation(rng, jnp.arange(N))
            mask = mask[:num_sampled]  # [num_sampled]

            # Sub-sample coords/targets
            coords_sub = coords_full[:, mask, :]    # [B, S, coord_dim]
            targets_sub = targets_full[:, mask, :]  # [B, S, out_dim]

            # (B) Replicate p, w from shape [1, L, ...] to [B, L, ...]
            p_batched = jnp.repeat(self.frozen_p, B, axis=0)  # [B, L, pos_dim(+ori)]
            w_batched = None
            if self.frozen_w is not None:
                w_batched = jnp.repeat(self.frozen_w, B, axis=0)  # [B, L, 1]

            # (C) The geometry latents a from the batch
            a_batched = batch['latents_c']  # [B, L, latent_dim(+cond)]

            # (D) Forward pass on sub-sampled coords
            preds = self.enf.apply(
                params['nef'],
                coords_sub,   # [B, S, coord_dim]
                p_batched,
                a_batched,
                w_batched
            )
            loss = jnp.mean((preds - targets_sub) ** 2)
            return loss, preds

        # Split RNG for mask generation
        rng, new_rng = jax.random.split(state.rng)
        (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, rng)

        updates, new_opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(params=new_params, opt_state=new_opt_state, rng=new_rng)
        return loss, new_state

    @partial(jax.jit, static_argnums=(0, 3))
    def eval_step(self, state, batch, subsample=True):
        """
        Evaluate the model. If use_full_domain=True, evaluate on all points.
        Otherwise, randomly sub-sample the domain.
        """
        rng, new_rng = jax.random.split(state.rng)
        coords_full = batch['input']
        targets_full = batch['output']
        B, N, _ = coords_full.shape

        if subsample:
            # Sub-sample
            num_sampled = self.cfg.training.num_sampled_points
            mask = jax.random.permutation(rng, jnp.arange(N))
            mask = mask[:num_sampled]
            coords_sub = coords_full[:, mask, :]
            targets_sub = targets_full[:, mask, :]
            
        else:
            # Use entire domain
            coords_sub = coords_full
            targets_sub = targets_full

        p_batched = jnp.repeat(self.frozen_p, B, axis=0)
        w_batched = None
        if self.frozen_w is not None:
            w_batched = jnp.repeat(self.frozen_w, B, axis=0)

        a_batched = batch['latents_c']
        preds = self.enf.apply(state.params['nef'], coords_sub, p_batched, a_batched, w_batched)
        loss = jnp.mean((preds - targets_sub) ** 2)

        new_state = state.replace(rng=new_rng)
        return loss, preds, new_state, coords_sub, targets_sub

    def evaluate_and_save(self, state, val_loader):
        """Base evaluation method that can be extended by subclasses."""
        val_losses = []
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        for batch in val_loader:
            loss, preds, new_state, coords, targets = self.eval_step(
                state, batch, subsample=False)
            self.state = new_state
            val_losses.append(loss)
            
            all_predictions.append(np.array(preds))
            all_inputs.append(np.array(coords))
            
            if 'output' in batch:
                all_targets.append(np.array(batch['output']))
        
        # Base results that all implementations need
        results = {
            'predictions': np.concatenate(all_predictions),
            'inputs': np.concatenate(all_inputs),
            'latent_positions': self.frozen_p
        }
        
        if all_targets:
            results['targets'] = np.concatenate(all_targets)
            
        # Allow subclasses to extend results
        self.extend_evaluation_results(results, all_targets)
            
        return float(np.mean(np.array(val_losses))), results
    
    def extend_evaluation_results(self, results, all_targets):
        """Hook for subclasses to extend evaluation results."""
        pass

    def train_model(self, num_epochs):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training loop
            train_losses = []
            for batch in self.train_loader:
                loss, new_state = self._train_step_loop(self.state, batch)
                self.state = new_state
                train_losses.append(loss)
            train_loss = float(np.mean(np.array(train_losses)))

            # Validation
            val_loss = self.validate(subsample=True)
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            if self.scheduler.should_reduce(val_loss):
                new_lr = self.scheduler.get_lr()
                print(f"  Reducing learning rate to {new_lr:.2e}")
                self.update_optimizer_state(new_lr)

            print(f"Epoch {epoch} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                if epoch > 10:
                    # Evaluate on the entire domain
                    val_loss_full, eval_results = self.evaluate_and_save(self.state, self.val_loader)
                    print(f"[Full-domain eval @ Epoch {epoch}] val_loss_full={val_loss_full:.6f}")
                    wandb.log({'val_loss_full_domain': val_loss_full, 'epoch': epoch})
                    # Save evaluation results
                    results_dir = os.path.join(self.cfg.experiment_dir, 'results')
                    os.makedirs(results_dir, exist_ok=True)
                    results_path = os.path.join(results_dir, 'val_predictions.npz')
                    np.savez(results_path, **eval_results)
                    print(f"Saved validation predictions to {results_path}")

                # Optionally save checkpoint, store best_state, etc.
                self.best_state = self.state
                print("  [Best val so far]")


        return self.state

    def _train_step_loop(self, state, batch):
        loss, new_state = self.train_step(state, batch)
        return loss, new_state

    def validate(self, subsample=True):
        """Compute validation loss (averaged over entire validation loader)."""
        val_losses = []
        for batch in self.val_loader:
            loss, preds, new_state, coords_sub, targets_sub = self.eval_step(self.state, batch, subsample=subsample)
            self.state = new_state  # update RNG
            val_losses.append(loss)
        return float(np.mean(np.array(val_losses)))

    def test(self, test_loader):
        """Memory-efficient testing that computes running statistics."""
        running_mse = 0.0
        total_samples = 0
        state = self.best_state
        
        for batch_idx, batch in enumerate(test_loader):
            coords_full = batch['input']      # shape [B, N, coord_dim]
            targets_full = batch['output']    # shape [B, N, out_dim]
            B = coords_full.shape[0]
            total_samples += B

            # Replicate p/w for the current batch size
            p_batched = jnp.repeat(self.frozen_p, B, axis=0)
            w_batched = None
            if self.frozen_w is not None:
                w_batched = jnp.repeat(self.frozen_w, B, axis=0)

            # Forward pass
            preds_full = self.enf.apply(
                state.params['nef'],
                coords_full,
                p_batched,
                batch['latents_c'],
                w_batched
            )

            # Update running MSE
            batch_mse = float(np.mean((np.array(preds_full) - np.array(targets_full)) ** 2))
            running_mse += batch_mse * B

            # Free memory explicitly
            del preds_full, coords_full, targets_full, p_batched, w_batched
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}, current MSE: {running_mse/total_samples:.6f}")

        final_mse = running_mse / total_samples
        return {'mse': final_mse}

    def update_optimizer_state(self, new_lr):
        """Update optimizer and create new optimization state."""
        self.current_lr = new_lr
        self.optimizer = optax.adam(new_lr)
        new_opt_state = self.optimizer.init(self.state.params)
        self.state = self.state.replace(
            opt_state=new_opt_state
        )
        
    # Optional checkpointing
    def save_checkpoint(self, path="output_enf_ckpt.pkl"):
        import pickle
        ckpt = {
            'params': self.state.params,
            'opt_state': self.state.opt_state,
            'rng': self.state.rng
        }
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, path="output_enf_ckpt.pkl"):
        import pickle
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        self.state = self.state.replace(
            params=ckpt['params'],
            opt_state=ckpt['opt_state'],
            rng=ckpt['rng']
        )
        print(f"[Checkpoint] Loaded from {path}")

class MetaSGDSteadyStateTrainer:
    class TrainState(struct.PyTreeNode):
        """Training state."""
        params: dict = struct.field(pytree_node=True)  # Contains nef, autodecoder, meta_sgd_lrs
        nef_opt_state: optax.OptState = struct.field(pytree_node=True)
        autodecoder_opt_state: optax.OptState = struct.field(pytree_node=True)
        meta_sgd_opt_state: optax.OptState = struct.field(pytree_node=True)
        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(self, config, nef, train_loader, val_loader, coords, seed=42):
        # Basic setup
        self.config = config
        self.nef = nef
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.coords = coords
        self.seed = seed
        self.epoch = 0
        self.global_step = 0
        self.metrics = {}

        
        coordinate_system = "cartesian"

        # Initialize outer autodecoder for learning template parameters
        self.outer_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=1,  # Single template
            num_latents=config.model.num_latents,
            latent_dim=config.model.latent_dim,
            spatial_dims=config.model.spatial_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.model.gaussian_window,
            coordinate_system=coordinate_system,
            bbox=config.model.latent_bbox
            
        )

        # Initialize inner autodecoder for per-sample adaptation
        self.inner_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=config.training.batch_size,  # One per sample
            num_latents=config.model.num_latents,
            latent_dim=config.model.latent_dim,
            spatial_dims=config.model.spatial_dim,
            num_pos_dims=nef.cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=nef.cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=config.model.gaussian_window,
            coordinate_system=coordinate_system,
            bbox=config.model.latent_bbox
        )

        # Setup checkpointing
        if self.config.logging.checkpoint:
            self.setup_checkpointing()

    def init_train_state(self):
        """Initialize training state with both autodecoders."""
        # Setup optimizers
        self.nef_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(self.config.optimizer.learning_rate_enf)
        )
        # Outer autodecoder optimizer - can be enabled/disabled via config
        self.autodecoder_opt = optax.adam(self.config.optimizer.learning_rate_codes)
        self.meta_sgd_opt = optax.adam(self.config.meta.learning_rate_meta_sgd)

        # Initialize random keys
        key = jax.random.PRNGKey(self.seed)
        key, nef_key, outer_key = jax.random.split(key, 3)

        # Initialize outer autodecoder parameters
        outer_params = self.outer_autodecoder.init(outer_key)
        p, a, window = self.outer_autodecoder.apply(outer_params)

        # Initialize meta-SGD learning rates
        meta_sgd_lrs = {
            'p_pos': jnp.zeros((1)) if self.config.model.freeze_lat_pos else jnp.ones((1)) * self.config.meta.inner_learning_rate_p,
            'a': jnp.ones((a.shape[-1])) * self.config.meta.inner_learning_rate_a,
            'gaussian_window': jnp.ones((1)) * self.config.meta.inner_learning_rate_window
        }

        # Initialize NeF
        sample_coords = jax.random.normal(nef_key, (1, 128, self.config.model.coord_dim))
        nef_params = self.nef.init(nef_key, sample_coords[:, :self.config.training.num_sampled_points], p, a, window)

        return self.TrainState(
            params={'nef': nef_params, 
                   'autodecoder': outer_params,
                   'meta_sgd_lrs': meta_sgd_lrs},
            nef_opt_state=self.nef_opt.init(nef_params),
            autodecoder_opt_state=self.autodecoder_opt.init(outer_params),
            meta_sgd_opt_state=self.meta_sgd_opt.init(meta_sgd_lrs),
            rng=key
        )
        
    def train_model(self, num_epochs):
        """Main training loop with validation every 5 epochs and checkpointing."""
        state = self.init_train_state()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            wandb.log({'epoch': epoch}, commit=False)
            
            # Training epoch
            loss_ep = 0
            for batch_idx, batch in enumerate(self.train_loader):
                loss, state = self.train_step(state, batch)
                loss_ep += loss
                self.global_step += 1

            # Compute mean training loss
            train_loss = loss_ep / len(self.train_loader)
            
            # Log training metrics
            metrics = {
                'train_loss_epoch': train_loss,
                'meta_lr_pos': float(state.params['meta_sgd_lrs']['p_pos'].mean()),
                'meta_lr_features': float(state.params['meta_sgd_lrs']['a'].mean()),
                'meta_lr_window': float(state.params['meta_sgd_lrs']['gaussian_window'].mean())
            }
            
            # Run validation only every 5 epochs
            if epoch % 5 == 0:
                val_loss = self.validate(state)
                metrics['val_loss'] = val_loss
                
                # Save checkpoint if we have best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"\nNew best validation loss at epoch {epoch}: {val_loss:.6f}")
                    self.save_checkpoint(state, is_best=True)
                    self.best_state = state
            
            # Log all metrics
            wandb.log(metrics, commit=True)
            
            # Print training progress
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.5f}")
            if epoch % 5 == 0:
                print(f"Val Loss: {val_loss:.5f}")
            print("Meta-learning rates:")
            print(f"  Pos: {float(state.params['meta_sgd_lrs']['p_pos'].mean()):.5f}")
            print(f"  Features: {float(state.params['meta_sgd_lrs']['a'].mean()):.5f}")
            print(f"  Window: {float(state.params['meta_sgd_lrs']['gaussian_window'].mean()):.5f}")
        
        return state

    @partial(jax.jit, static_argnums=(0,))
    def inner_loop(self, outer_params, outer_state, batch):
        """Inner loop optimization using inner autodecoder."""
        # Unpack batch - now expecting dictionary format
        coords = batch['input']  # Shape: (batch_size, num_points, coord_dim)
        img = batch['output']    # Shape: (batch_size, num_points, out_dim)

        # Generate coordinate masks - now per batch item
        batch_size = coords.shape[0]
        num_points = coords.shape[1]
        
        # Generate mask for coordinate sampling
        mask = jax.random.permutation(
            outer_state.rng,
            jnp.broadcast_to(
                jnp.arange(num_points)[:, jnp.newaxis],
                (num_points, self.config.training.num_inner_steps + 1)
            ),
            independent=True,
        )
        mask = mask[:self.config.training.num_sampled_points, :]

        # Initialize inner autodecoder with broadcast outer parameters
        inner_autodecoder_params = jax.tree_map(
            lambda p: jnp.repeat(p, batch_size, axis=0),
            outer_params['autodecoder']
        )

        def loss_fn(params, masked_coords, masked_img):
            p, a, window = self.inner_autodecoder.apply(params['autodecoder'])
            out = self.nef.apply(params['nef'], masked_coords, p, a, window)
            return jnp.mean((out - masked_img) ** 2)

        inner_state = outer_state.replace(
            params={'nef': outer_params['nef'], 
                   'autodecoder': inner_autodecoder_params,
                   'meta_sgd_lrs': outer_params['meta_sgd_lrs']}
        )

        # Inner optimization loop
        for inner_step in range(self.config.training.num_inner_steps):
            # Select masked coordinates for each batch item
            masked_coords = jax.vmap(lambda x: x[mask[:, inner_step]])(coords)
            masked_img = jax.vmap(lambda x: x[mask[:, inner_step]])(img)

            # Get gradients
            inner_grad = jax.grad(loss_fn)(
                inner_state.params,
                masked_coords=masked_coords,
                masked_img=masked_img,
            )
            
            # Optionally zero out gaussian window param updates.
            if 'gaussian_window' in inner_grad['autodecoder']['params']:
                inner_grad['autodecoder']['params']['gaussian_window'] = jnp.zeros_like(
                    inner_grad['autodecoder']['params']['gaussian_window'])
            if 'p_pos' in inner_grad['autodecoder']['params']:
                inner_grad['autodecoder']['params']['p_pos'] = jnp.zeros_like(
                    inner_grad['autodecoder']['params']['p_pos'])


            # Apply meta-learning rates based on config
            inner_updates = jax.tree_util.tree_map_with_path(
                lambda path, grad: -inner_state.params['meta_sgd_lrs'][path[-1].key] * grad,
                inner_grad['autodecoder']
            )

            # Update parameters
            updated_params = optax.apply_updates(inner_state.params['autodecoder'], inner_updates)
            inner_state = inner_state.replace(
                params={'nef': inner_state.params['nef'], 
                       'autodecoder': updated_params,
                       'meta_sgd_lrs': inner_state.params['meta_sgd_lrs']}
            )

        # Final loss computation with masked coordinates
        masked_coords = jax.vmap(lambda x: x[mask[:, -1]])(coords)
        masked_img = jax.vmap(lambda x: x[mask[:, -1]])(img)

        final_loss = loss_fn(
            inner_state.params,
            masked_coords=masked_coords,
            masked_img=masked_img,
        )

        return final_loss, inner_state

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch):
        """Training step updating both NeF and outer autodecoder."""
        # Split random key
        key, new_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_key)

        # Get gradients
        (loss, _), grads = jax.value_and_grad(self.inner_loop, has_aux=True)(
            outer_state.params, outer_state, batch
        )

        # Update NeF parameters
        nef_updates, nef_opt_state = self.nef_opt.update(grads['nef'], state.nef_opt_state, state.params['nef'])
        nef_params = optax.apply_updates(state.params['nef'], nef_updates)

        # Update outer autodecoder if enabled
        if self.config.optimizer.learning_rate_codes > 0:
            autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(
                grads['autodecoder'], state.autodecoder_opt_state
            )
            autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)
        else:
            autodecoder_params = state.params['autodecoder']
            autodecoder_opt_state = state.autodecoder_opt_state

        # Update meta-SGD learning rates
        meta_updates, meta_opt_state = self.meta_sgd_opt.update(grads['meta_sgd_lrs'], state.meta_sgd_opt_state)
        meta_sgd_lrs = optax.apply_updates(state.params['meta_sgd_lrs'], meta_updates)
        meta_sgd_lrs = jax.tree_map(lambda x: jnp.clip(x, 1e-6, 10), meta_sgd_lrs)

        return loss, state.replace(
            params={'nef': nef_params,
                   'autodecoder': autodecoder_params,
                   'meta_sgd_lrs': meta_sgd_lrs},
            nef_opt_state=nef_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
            meta_sgd_opt_state=meta_opt_state,
            rng=new_key
        )

    def validate(self, state):
        """Validation loop."""
        val_losses = []
        for batch in self.val_loader:
            loss, _ = self.inner_loop(state.params, state, batch)
            val_losses.append(loss)
        return np.mean(val_losses)

    # First, let's modify the test method in MetaSGDSteadyStateTrainer to return latents

    # Modified test method for MetaSGDSteadyStateTrainer to accept override_latents
    def test(self, test_loader, override_latents=None):
        self.state = self.best_state
        """Testing loop with optional latent override."""
        test_losses = []
        all_latents_p = []
        all_latents_c = []
        all_predictions = []
        all_targets = []
        
        for batch in test_loader:
            if override_latents is not None:
                # Use provided latents instead of computing them
                p = override_latents['latents_p']
                a = override_latents['latents_c']
                # Still need to compute window
                _, _, window = self.inner_autodecoder.apply(self.state.params['autodecoder'])
            else:
                # Normal latent computation through inner loop
                loss, inner_state = self.inner_loop(self.state.params, self.state, batch)
                test_losses.append(loss)
                p, a, window = self.inner_autodecoder.apply(inner_state.params['autodecoder'])
            
            # Generate predictions
            pred = self.nef.apply(
                self.state.params['nef'],
                batch['input'],
                p,
                a,
                window
            )
            
            # Store results
            all_latents_p.append(np.array(p))
            all_latents_c.append(np.array(a))
            all_predictions.append(np.array(pred))
            if 'output' in batch:
                all_targets.append(np.array(batch['output']))
        
        results = {
            'latents_p': np.concatenate(all_latents_p),
            'latents_c': np.concatenate(all_latents_c),
            'predictions': np.concatenate(all_predictions),
        }
        
        if test_losses:
            results['loss'] = np.mean(test_losses)
        if all_targets:
            results['targets'] = np.concatenate(all_targets)
        
        return results

    def setup_checkpointing(self):
        """Setup checkpointing directory and manager."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if hasattr(self.config, 'experiment_dir'):
            # Convert to absolute path if not already
            self.experiment_dir = os.path.abspath(self.config.experiment_dir)
            self.checkpoint_dir = os.path.abspath(os.path.join(self.experiment_dir, "checkpoints"))
        else:
            # Create new experiment directory with absolute paths
            self.experiment_dir = os.path.abspath(os.path.join(
                self.config.logging.log_dir, 
                f"experiment_{timestamp}"
            ))
            self.checkpoint_dir = os.path.abspath(os.path.join(
                self.experiment_dir, 
                "checkpoints"
            ))
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.abspath(os.path.join(self.experiment_dir, "config.yaml"))
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self.config, f=f)
        
        # Setup checkpoint manager with absolute paths
        self.checkpoint_options = ocp.CheckpointManagerOptions(
            max_to_keep=self.config.logging.keep_n_checkpoints,
            create=True,
            cleanup_tmp_directories=True
        )
        
        self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        
        # Ensure checkpoint directory is absolute
        self.checkpoint_manager = ocp.CheckpointManager(
            directory=os.path.abspath(self.checkpoint_dir),
            checkpointers=self.checkpointer,
            options=self.checkpoint_options,
            metadata=None
        )
        
        print(f"\nExperiment directory (abs): {os.path.abspath(self.experiment_dir)}")
        print(f"Checkpoints directory (abs): {os.path.abspath(self.checkpoint_dir)}")
        print(f"Saved config to (abs): {os.path.abspath(config_path)}")
        
        return self.experiment_dir

    def save_checkpoint(self, state, is_best=False):
        """Save checkpoint."""
        if not self.config.logging.checkpoint:
            return

        save_args = orbax_utils.save_args_from_target(state)
        
        # Save current state
        checkpoint_name = f"epoch_{self.epoch}"
        if is_best:
            checkpoint_name += "_best"
            
        print(f"\nSaving checkpoint: {checkpoint_name}")
        self.checkpoint_manager.save(
            self.epoch,
            state,
            save_kwargs={'save_args': save_args}
        )

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint from path or latest available."""
        if checkpoint_path is not None:
            self.checkpoint_dir = checkpoint_path
            self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
            self.checkpoint_manager = ocp.CheckpointManager(
                directory=checkpoint_path,
                checkpointers=self.checkpointer,
                options=self.checkpoint_options
            )

        # Load latest checkpoint
        available_steps = self.checkpoint_manager.all_steps()
        if not available_steps:
            print("No checkpoints found.")
            return self.init_train_state()
        
        step_to_restore = max(available_steps)
        print(f"\nLoading checkpoint from step {step_to_restore}")
        
        # Initialize empty state for structure
        empty_state = self.init_train_state()
        
        # Restore the checkpoint
        loaded_state = self.checkpoint_manager.restore(
            step_to_restore,
            items=empty_state,
            restore_kwargs={'restore_args': orbax_utils.restore_args_from_target(empty_state)}
        )
        
        # Store the loaded state
        self.state = loaded_state
        
        # Print info about loaded state
        print("\nLoaded checkpoint info:")
        print(f"NeF params shape: {jax.tree_map(lambda x: x.shape, loaded_state.params['nef'])}")
        print(f"Meta-SGD learning rates:")
        for k, v in loaded_state.params['meta_sgd_lrs'].items():
            print(f"  {k}: min={float(v.min()):.6f}, max={float(v.max()):.6f}, mean={float(v.mean()):.6f}")
        
        return loaded_state