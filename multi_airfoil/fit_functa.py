from utils_functa.load_models import create_inr_instance
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
from utils_functa.dataset import Struct, VariantSDFDataset
import os
import airfrans as af
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data, Dataset, DataLoader
import wandb
import matplotlib.pyplot as plt
import json 

# adapted from https://github.com/EmilienDupont/coinpp/blob/main/coinpp/metalearning.py and from https://github.com/LouisSerrano/coral


def train_inr(cfg, train_dataset, val_dataset):
    # Accessing parameters using dot notation
    batch_size = cfg.optim.batch_size
    epochs = cfg.optim.epochs
    inner_steps = cfg.optim.inner_steps
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    input_dim = cfg.inr_in.input_dim
    output_dim = cfg.inr_in.output_dim
    latent_dim = cfg.inr_in.latent_dim
    num_points = cfg.optim.num_points

    # Create INR instance
    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    alpha_in = nn.Parameter(torch.Tensor([lr_code]).cuda())

    optimizer_in = torch.optim.AdamW(
        [
            {"params": inr_in.parameters()},
            {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    best_loss = np.inf
    train_loss_history, test_loss_history = [], []
    
    # Initialize wandb
    wandb.init(project='multielement_sdf', config=cfg)

    for step in tqdm(range(epochs), desc="Training INR"):
        train_loss_in, test_loss_in = 0, 0
        fit_train_mse_in, fit_test_mse_in = 0, 0
        use_rel_loss = step % 10 == 0
        
        # Dynamic subsampling for training
        train_loader = DataLoader(
            subsample_dataset(train_dataset, num_points),
            batch_size=batch_size,
            shuffle=True,
        )

        # Training loop
        for graph in train_loader:
            n_samples = graph.num_graphs
            inr_in.train()
            graph.to("cuda")
            graph.modulations = torch.zeros(batch_size, latent_dim, device="cuda")

            outputs = graph_outer_step(
                inr_in,
                graph,
                inner_steps,
                alpha_in,
                is_train=True,
            )

            optimizer_in.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
            optimizer_in.step()
            loss = outputs["loss"].cpu().detach()
            fit_train_mse_in += loss.item() * n_samples
            z0 = outputs["modulations"].detach()

        train_loss_in = fit_train_mse_in / len(train_dataset)
        train_loss_history.append({"epoch": step, "loss": train_loss_in})
        print("Train Loss", train_loss_in)
        
        # Log training loss to wandb
        wandb.log({"epoch": step, "train_loss": train_loss_in})

        # Validation loop
        if step % 5 == 0:  # Adjust as per your validation frequency
            # Dynamic subsampling for training
            val_loader = DataLoader(
                subsample_dataset(val_dataset, num_points),
                batch_size=batch_size,
                shuffle=True,
            )

            for substep, graph in enumerate(val_loader):
                n_samples = len(graph)
                inr_in.train()

                graph.modulations = torch.zeros(batch_size, latent_dim)
                graph = graph.cuda()

                outputs = graph_outer_step(
                    inr_in,
                    graph,
                    inner_steps,
                    alpha_in,
                    is_train=False,
                )
                loss = outputs["loss"].cpu().detach()
                fit_test_mse_in += loss.item() * n_samples
                z0 = outputs["modulations"].detach()

            test_loss_in = fit_test_mse_in / len(val_dataset)
            test_loss_history.append({"epoch": step, "loss": test_loss_in})
            print("Test Loss", test_loss_in)
            
            # Log validation loss to wandb
            wandb.log({"epoch": step, "val_loss": test_loss_in})

            if test_loss_in < best_loss:
                best_loss = test_loss_in
                best_model_state = {
                    "cfg": cfg,
                    "epoch": step,
                    "inr_in": inr_in.state_dict(),
                    "optimizer_inr_in": optimizer_in.state_dict(),
                    "alpha_in": alpha_in,
                }
                
    print("Extracting final modulations and validation results...")
    
    # Extract training modulations
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
    )
    
    # Extract validation modulations and predictions
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )
    
    val_results = {
        'inputs': [],
        'predictions': [],
        'targets': []
    }
    
    for graph in val_loader:
        inr_in.train()
        graph.to("cuda")
        graph.modulations = torch.zeros(len(graph), latent_dim, device="cuda")
        
        outputs = graph_outer_step(
            inr_in,
            graph,
            inner_steps,
            alpha_in,
            is_train=False,
        )
        
        
        # Store inputs, predictions, and targets
        val_results['inputs'].append(graph.input.cpu().numpy())
        predictions = inr_in.modulated_forward(graph.input, outputs["modulations"]).detach()
        val_results['predictions'].append(predictions.cpu().numpy())
        val_results['targets'].append(graph.output.cpu().numpy())
    
    
    # Convert lists to numpy arrays
    val_results['inputs'] = np.concatenate(val_results['inputs'], axis=0)
    val_results['predictions'] = np.concatenate(val_results['predictions'], axis=0)
    val_results['targets'] = np.concatenate(val_results['targets'], axis=0)
    
    # Save modulations
    save_dir = os.path.join(cfg.output_dir, 'functa_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save validation results
    np.savez(
        os.path.join(save_dir, f'val_results_{str(latent_dim)}.npz'),
        **val_results
    )
    
    print(f"Saved validation results with shapes:")
    print(f"  inputs: {val_results['inputs'].shape}")
    print(f"  predictions: {val_results['predictions'].shape}")
    print(f"  targets: {val_results['targets'].shape}")

    return best_model_state

def subsample_dataset(dataset, num_points):
    subsampled_dataset = []
    for data in dataset:
        total_points = data.num_nodes
        if total_points > num_points:
            subsample_indices = np.random.choice(total_points, num_points, replace=False)
            new_data = Data(
                input=data.input[subsample_indices],
                pos=data.pos[subsample_indices],
                output=data.output[subsample_indices],
                is_airfoil=data.is_airfoil[subsample_indices],
            )
            subsampled_dataset.append(new_data)
        else:
            subsampled_dataset.append(data)
    return subsampled_dataset

def graph_inner_loop(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_steps,
    inner_lr,
    is_train=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.


    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        fitted_modulations = graph_inner_loop_step(
            func_rep,
            fitted_modulations,
            coords,
            features,
            batch_index,
            inner_lr,
            is_train,
        )

    return fitted_modulations


def graph_inner_loop_step(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
):
    """Performs a single inner loop step."""
    detach = False
    batch_size = modulations.shape[0]
    loss = 0
    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch

        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        loss = ((features_recon - features) ** 2).mean() * batch_size
        
        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]

    # Perform single gradient descent step
    return modulations - inner_lr * grad


def graph_outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    is_train=False,
    detach_modulations=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    func_rep.zero_grad()
    batch_size = len(graph)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.zeros_like(graph.modulations).requires_grad_()
    coords = graph.input
    features = graph.output

    # Run inner loop
    modulations = graph_inner_loop(
        func_rep,
        modulations,
        coords,
        features,
        graph.batch,
        inner_steps,
        inner_lr,
        is_train,
    )

    if detach_modulations:
        modulations = modulations.detach()  # 1er ordre

    loss = 0
    batch_size = modulations.shape[0]

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations[graph.batch])
        loss = ((features_recon - features) ** 2).mean()

    outputs = {
        "loss": loss,
        "modulations": modulations,
    }

    return outputs


 
if __name__ == "__main__":
    
    with open('config_functa.json', 'r') as f:
        cfg = json.load(f)
    if isinstance(cfg, dict):
        # cfg = OmegaConf.create(cfg)
        cfg = Struct(cfg)

        
    # Load the variants dataset from geom_db.npy
    variants = np.load("geom_db.npy", allow_pickle=True).item()
    print(f"Loaded {len(variants)} variants.")

    # Split variants into train and validation (e.g., 80/20 split)
    variant_keys = list(variants.keys())
    np.random.shuffle(variant_keys)
    split = int(0.8 * len(variant_keys))
    train_keys = variant_keys[:split]
    val_keys = variant_keys[split:]
    train_variants = {k: variants[k] for k in train_keys}
    val_variants = {k: variants[k] for k in val_keys}

    # Create the PyG datasets for the variants. (Normalization is computed on train set.)
    train_dataset = VariantSDFDataset(train_variants, is_train=True)
    val_dataset = VariantSDFDataset(val_variants, is_train=False, coef_norm=train_dataset.coef_norm)

    # Train the INR on the variant dataset.
    best_model_state = train_inr(cfg, train_dataset, val_dataset)
    
    wandb.finish()