import numpy as np
import pyvista as pv
import os.path as osp
import os
import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import airfrans as af    
import numpy as np
import jax.numpy as jnp
import os
import einops

class JAXTransonicRAE:
    def __init__(self, data_directory, target_field, num_points=None, global_norm=False, coef_norm=None, add_cond_in=False):
        self.data_directory = data_directory
        self.target_field = target_field
        self.num_points = num_points
        self.global_norm = global_norm
        self.coef_norm = coef_norm or {'mean': None, 'std': None, 'min': None, 'max': None}
        self.add_cond_in = add_cond_in

        print("Processing dataset...")
        self.dataset = self.process_data()

    def process_data(self):
        dataset = []
        print('Loading raw data')
        db_random = np.load(os.path.join(self.data_directory, 'db_random.npy'), allow_pickle=True).item()
        db_cyc = np.load(os.path.join(self.data_directory, 'db_cyc.npy'), allow_pickle=True).item()

        # Merge db_random and db_cyc
        db = {key: np.concatenate((db_random[key], db_cyc[key]), axis=0) for key in ['Pressure','Xcoordinate','Ycoordinate','Vinf','Alpha','idx']}
        print('Raw data loaded, normalizing data')

        alfa  = 2 * (db['Alpha'] - db['Alpha'].min()) / (db['Alpha'].max() - db['Alpha'].min()) - 1
        Vinf  = 2 * (db['Vinf'] - db['Vinf'].min()) / (db['Vinf'].max() - db['Vinf'].min()) - 1

        # Normalize target field
        if self.target_field in db:
            target_data = db[self.target_field]
            mean_out, std_out = target_data.mean(), target_data.std()
            db[self.target_field] = (target_data - mean_out) / std_out
            self.coef_norm.update({'mean': mean_out, 'std': std_out})

        if self.global_norm:
            min_out, max_out = target_data.min(), target_data.max()
            db[self.target_field] = (target_data - min_out) / (max_out - min_out)
            self.coef_norm.update({'min': min_out, 'max': max_out})
            
        for idx in range(len(db['idx'])):
            if db['Vinf'][idx] / 347 >= 0.2:
                X_coord = 2*(db['Xcoordinate'][idx] - db['Xcoordinate'][idx].min()) / (db['Xcoordinate'][idx].max() - db['Xcoordinate'][idx].min()) - 1
                Y_coord = 2*(db['Ycoordinate'][idx] - db['Ycoordinate'][idx].min()) / (db['Ycoordinate'][idx].max() - db['Ycoordinate'][idx].min()) - 1
                pos = np.stack((X_coord, Y_coord), axis=1)
                output = db[self.target_field][idx]
                cond_norm = np.array([[alfa[idx], Vinf[idx]]] * pos.shape[0])
                cond = np.array([[alfa[idx], Vinf[idx]]]) 
                input_data = pos
                 
                input_data = pos.reshape(*pos.shape)
                output = output.reshape(*output.shape,1)
                
                dataset.append({'pos': pos, 'input': input_data, 'output': output, 'cond': cond})
                
        return dataset

    def create_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        if seed is not None:
            np.random.seed(seed)

        total_samples = len(self.dataset)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(val_ratio * total_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        self.train_dataset = [self.dataset[i] for i in train_indices]
        self.val_dataset = [self.dataset[i] for i in val_indices]
        self.test_dataset = [self.dataset[i] for i in test_indices]

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def subsample_dataset(dataset, num_points):
        subsampled_dataset = []
        for data_entry in dataset:
            if len(data_entry['input']) > num_points:
                subsample_indices = np.random.choice(len(data_entry['input']), num_points, replace=False)
                subsampled_data = {key: data_entry[key][subsample_indices] for key in data_entry}
                subsampled_dataset.append(subsampled_data)
            else:
                subsampled_dataset.append(data_entry)
        return subsampled_dataset


class AirfransDataset:
    def __init__(self, data_list, data_names, target_field='pressure', is_train=True, coef_norm=None, num_points=None):
        """
        Initialize the AirfransDataset class with pre-loaded data.
        
        Args:
            data_list: list of np.arrays containing the raw data
            data_names: list of data names
            target_field: str, one of ['velocity', 'pressure', 'turbulent_viscosity', 'implicit_distance']
            is_train: bool, whether this is training data (to compute normalization)
            coef_norm: dict, normalization coefficients (required if is_train=False)
            num_points: int, number of points to subsample (if None, use all points)
        """
        self.data_list = data_list
        self.data_names = data_names
        self.target_field = target_field
        self.num_points = num_points
        self.is_train = is_train
        
        # Define feature indices
        self.feature_indices = {
            'position': slice(0, 2),  # x, y coordinates
            'inlet_velocity': slice(2, 4),  # velocity components
            'distance': 4,  # distance to airfoil
            'normals': slice(5, 7),  # normal components
            'velocity': slice(7, 9),  # target velocity components
            'all_outputs': slice(7, 11),
            'pressure': 9,  # target pressure
            'turbulent_viscosity': 10,  # target turbulent viscosity
            'is_airfoil': 11  # boolean flag for airfoil points
        }
        
        if is_train:
            # Compute normalization parameters from data
            print("Computing normalization parameters from training data...")
            self.compute_norm_params(data_list)
        else:
            # Use provided normalization parameters
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train=False")
            print("Using provided normalization parameters...")
            self.coef_norm = coef_norm
            self.pos_norm = coef_norm['pos_norm']
            self.inlet_norm = coef_norm['inlet_norm']
            if self.target_field in ['pressure','turbulent_viscosity','velocity','all_outputs']:
                self.implicit_norm = coef_norm['implicit_norm']
        
        # Process data
        print("Processing dataset...")
        self.processed_dataset = self.process_data(data_list)

    def compute_norm_params(self, data_list):
        """Compute normalization parameters."""
        self.coef_norm = {}
        
        # For target field
        target_indices = {
            'velocity': slice(7, 9),
            'pressure': 9,
            'turbulent_viscosity': 10,
            'all_outputs': slice(7, 11),
            'implicit_distance': 4
        }
        target_idx = target_indices[self.target_field]
        
        # Calculate target normalization
        if isinstance(target_idx, slice):
            all_targets = np.concatenate([data[:, target_idx] for data in data_list])
        else:
            all_targets = np.concatenate([data[:, target_idx:target_idx+1] for data in data_list])
        
        self.coef_norm['mean'] = all_targets.mean(axis=0)
        self.coef_norm['std'] = all_targets.std(axis=0)
        
        # Calculate position normalization
        all_positions = np.concatenate([data[:, :2] for data in data_list])
        self.pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        self.coef_norm['pos_norm'] = self.pos_norm
        
        # Calculate inlet velocity normalization
        all_inlet_vel = np.concatenate([data[:, 2:4] for data in data_list])
        self.inlet_norm = {
            'min': all_inlet_vel.min(axis=0),
            'max': all_inlet_vel.max(axis=0)
        }
        self.coef_norm['inlet_norm'] = self.inlet_norm
        
        # Calculate implicit distance normalization if needed
        if self.target_field in ['pressure','turbulent_viscosity','velocity','all_outputs']:
            all_implicit_dist = np.concatenate([data[:, 4:5] for data in data_list])
            self.implicit_norm = {
                'min': all_implicit_dist.min(axis=0),
                'max': all_implicit_dist.max(axis=0)
            }
            self.coef_norm['implicit_norm'] = self.implicit_norm

        print("\nNormalization parameters computed:")
        print(f"Target mean: {np.array2string(self.coef_norm['mean'], precision=4, separator=', ')}")
        print(f"Target std: {np.array2string(self.coef_norm['std'], precision=4, separator=', ')}")
        return self.coef_norm

    def process_data(self, dataset_list):
        """Process the raw data into the required format using global normalization."""
        dataset = []
        
        target_indices = {
            'all_outputs': slice(7, 11),
            'velocity': slice(7, 9),
            'pressure': 9,
            'turbulent_viscosity': 10,
            'implicit_distance': 4
        }
        target_idx = target_indices[self.target_field]
        
        # Process each simulation
        for i,sim_data in enumerate(dataset_list):
            if self.data_names[i] == 'airFoil2D_SST_50.077_-4.416_2.834_4.029_1.0_5.156':
                print("Skipping sample airFoil2D_SST_50.077_-4.416_2.834_4.029_1.0_5.156")
                continue
                
            # Extract and normalize positions
            pos = sim_data[:, :2]
            pos = 2 * (pos - self.pos_norm['min']) / (self.pos_norm['max'] - self.pos_norm['min']) - 1
            
            # Extract and normalize inlet velocity
            inlet_vel = sim_data[:, 2:4]
            inlet_vel_norm = 2 * (inlet_vel - self.inlet_norm['min']) / (self.inlet_norm['max'] - self.inlet_norm['min']) - 1
            
            normals = sim_data[:, 5:7]
            # Prepare input data based on target field
            if self.target_field in ['pressure','turbulent_viscosity','velocity','all_outputs']:
                # Include normalized implicit distance for pressure prediction
                implicit_dist = sim_data[:, 4:5]
                implicit_dist_norm = 2 * (implicit_dist - self.implicit_norm['min']) / (self.implicit_norm['max'] - self.implicit_norm['min']) - 1
                input_data = np.concatenate([pos, implicit_dist_norm], axis=1)
            else:
                input_data = pos
            
            # Extract and normalize target data
            if isinstance(target_idx, slice):
                output = sim_data[:, target_idx]
            else:
                output = sim_data[:, target_idx:target_idx+1]
            
            # Normalize target using mean and std
            output = (output - self.coef_norm['mean']) / self.coef_norm['std']
            
            # Create conditional input (mean inlet velocity)
            cond = np.mean(inlet_vel_norm, axis=0, keepdims=True)
            
            # Create data entry
            data_entry = {
                'pos': pos,
                'input': input_data,
                'output': output,
                'cond': cond,
                'is_airfoil': sim_data[:, -1],
                'normals': normals
            }
            
            dataset.append(data_entry)
        
        # Subsample if requested
        if self.num_points is not None:
            dataset = self.subsample_dataset(dataset, self.num_points)
            
        return dataset

    @staticmethod
    def subsample_dataset(dataset, num_points):
        """Subsample the dataset to a fixed number of points per simulation."""
        subsampled_dataset = []
        for data_entry in dataset:
            total_points = len(data_entry['input'])
            
            if total_points > num_points:
                # Generate indices without replacement
                subsample_indices = np.random.choice(total_points, num_points, replace=False)
                
                # Create new data entry with subsampled data
                new_entry = data_entry.copy()
                new_entry['input'] = data_entry['input'][subsample_indices]
                new_entry['output'] = data_entry['output'][subsample_indices]
                new_entry['pos'] = data_entry['pos'][subsample_indices]
                new_entry['is_airfoil'] = data_entry['is_airfoil'][subsample_indices]
                new_entry['normals'] = data_entry['normals'][subsample_indices]
                
                subsampled_dataset.append(new_entry)
            else:
                subsampled_dataset.append(data_entry)
                
        return subsampled_dataset

    def __getitem__(self, index):
        """Get a specific data entry."""
        return self.processed_dataset[index]

    def __len__(self):
        """Get dataset length."""
        return len(self.processed_dataset)
    
    
class ElasticityDataset:
    def __init__(self, data_directory, ntrain=1000, ntest=200, target_field='sigma'):
        """
        Initialize the ElasticityDataset class.
        
        Args:
            data_directory: str, path to data directory
            ntrain: int, number of training samples
            ntest: int, number of test samples
            target_field: str, either 'sigma' or 'xy'
        """
        if target_field not in ['sigma', 'xy']:
            raise ValueError("target_field must be either 'sigma' or 'xy'")
            
        self.data_directory = data_directory
        self.target_field = target_field
        self.ntrain = ntrain
        self.ntest = ntest
        
        print(f"Loading and processing data for target field: {target_field}...")
        self.load_and_process_data()

    def load_and_process_data(self):
        """Load and process data, creating train and test splits."""
        # Load raw data
        PATH_Sigma = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_sigma_10.npy")
        PATH_XY = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_XY_10.npy")
        PATH_rr = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_rr_10.npy")
        PATH_theta = os.path.join(self.data_directory, "Meshes-20241205T151153Z-001/Meshes/Random_UnitCell_theta_10.npy")

        input_rr = torch.tensor(np.load(PATH_rr), dtype=torch.float).permute(1, 0)
        input_sigma = torch.tensor(np.load(PATH_Sigma), dtype=torch.float).permute(1, 0).unsqueeze(-1)
        input_xy = torch.tensor(np.load(PATH_XY), dtype=torch.float).permute(2, 0, 1)
        input_theta = torch.tensor(np.load(PATH_theta), dtype=torch.float)

        # Split data
        train_rr = input_rr[:self.ntrain]
        test_rr = input_rr[-self.ntest:]
        train_theta = input_theta[:self.ntrain]
        test_theta = input_theta[-self.ntest:]
        train_sigma = input_sigma[:self.ntrain]
        test_sigma = input_sigma[-self.ntest:]
        train_xy = input_xy[:self.ntrain]
        test_xy = input_xy[-self.ntest:]

        # Get average grid
        average_grid = train_xy.mean(0)
        grid_tr = einops.repeat(average_grid, '... -> b ...', b=train_xy.shape[0])
        grid_te = einops.repeat(average_grid, '... -> b ...', b=test_xy.shape[0])

        # Compute normalization parameters and normalize based on target field
        if self.target_field == 'sigma':
            # Normalize sigma values
            self.sigma_mean = 0
            self.sigma_std = train_sigma.std()
            train_output = (train_sigma - self.sigma_mean) / self.sigma_std
            test_output = (test_sigma - self.sigma_mean) / self.sigma_std
            
            # Use xy as input when target is sigma
            train_input = (train_xy-0.5)*2
            test_input = (test_xy-0.5)*2
            
        else:  # target_field == 'xy'
            # Normalize xy coordinates
            self.xy_mean = train_xy.mean(dim=(0, 1))
            self.xy_std = train_xy.std(dim=(0, 1))
            train_output = (train_xy - self.xy_mean) / self.xy_std
            test_output = (test_xy - self.xy_mean) / self.xy_std
            
            # Use grid as input when target is xy
            train_input = grid_tr
            test_input = grid_te

        # Create train dataset
        self.train_dataset = []
        for i in range(len(train_xy)):
            self.train_dataset.append({
                'pos': grid_tr[i],  # pos always uses the grid
                'input': train_input[i],  # input varies based on target_field
                'output': train_output[i],
                'cond': train_rr[i],
                'theta': train_theta[i]
            })

        # Create test dataset
        self.test_dataset = []
        for i in range(len(test_xy)):
            self.test_dataset.append({
                'pos': grid_te[i],  # pos always uses the grid
                'input': test_input[i],  # input varies based on target_field
                'output': test_output[i],
                'cond': test_rr[i],
                'theta': test_theta[i]
            })

    def get_normalization_stats(self):
        """Get normalization statistics."""
        if self.target_field == 'sigma':
            return {'mean': self.sigma_mean, 'std': self.sigma_std}
        else:
            return {'mean': self.xy_mean, 'std': self.xy_std}
        


class JAXDataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))

    def __iter__(self):
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indices)
        for start_idx in range(0, len(self.indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            batch_data = [self.data[i] for i in batch_indices if self.data[i] is not None]  # Ensure valid data
            batch_dict = {key: np.stack([d[key] for d in batch_data]) for key in batch_data[0].keys()}
            yield batch_dict

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
def setup_datasets(cfg, num_points=100000):
    """Setup all required datasets based on configuration.
    
    Args:
        cfg: Configuration object containing dataset settings
        num_points: Number of points to sample (only used for airfrans dataset)
    
    Returns:
        dict: Dictionary containing train and test dataset pairs for each field
    """
    if cfg.dataset.name == 'elasticity':
        # Setup input (xy) dataset
        dataset_in = ElasticityDataset(
            data_directory=cfg.dataset.root_path,
            ntrain=1000,
            ntest=200,
            target_field='xy')
        train_dataset_in, test_dataset_in = dataset_in.train_dataset, dataset_in.test_dataset
        
        # Setup output (sigma) dataset
        dataset_out = ElasticityDataset(
            data_directory=cfg.dataset.root_path,
            ntrain=1000,
            ntest=200,
            target_field='sigma')
        train_dataset_out, test_dataset_out = dataset_out.train_dataset, dataset_out.test_dataset
        
        return {
            'in': (train_dataset_in, test_dataset_in),
            'out': (train_dataset_out, test_dataset_out)
        }
        
    elif cfg.dataset.name == 'airfrans':
        # Load the full dataset
        train_data, train_names = af.dataset.load(root=cfg.dataset.root_path, task='full', train=True)
        test_data, test_names = af.dataset.load(root=cfg.dataset.root_path, task='full', train=False)
        
        # Create pressure datasets
        train_dataset_p = AirfransDataset(
            data_list=train_data,
            data_names=train_names,
            target_field=cfg.dataset.target_fields.pressure,
            is_train=True,
            num_points=num_points
        )
        test_dataset_p = AirfransDataset(
            data_list=test_data,
            data_names=test_names,
            target_field=cfg.dataset.target_fields.pressure,
            is_train=False,
            coef_norm=train_dataset_p.coef_norm,
            num_points=num_points
        )
        
        # Create geometry datasets
        train_dataset_g = AirfransDataset(
            data_list=train_data,
            data_names=train_names,
            target_field=cfg.dataset.target_fields.geometry,
            is_train=True,
            num_points=num_points
        )
        test_dataset_g = AirfransDataset(
            data_list=test_data,
            data_names=test_names,
            target_field=cfg.dataset.target_fields.geometry,
            is_train=False,
            coef_norm=train_dataset_g.coef_norm,
            num_points=num_points
        )
        
        return {
            'pressure': (train_dataset_p, test_dataset_p),
            'geometry': (train_dataset_g, test_dataset_g)
        }
        
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}. Expected 'elasticity' or 'airfrans'")
    
    
class OutputLatentDataset:
    """
    Dataset that merges:
      - The base dataset (pressure/elasticity with coords, PDE output, etc.)
      - The geometry latents from the geometry ENF (p, a)
      - Optionally conditions, appended to latents_c if desired
    
    Normalizes full latents_c (including conditions) to standard Gaussian distribution.
    For validation set, uses pre-computed normalization statistics from training set.
    """
    def __init__(self, base_dataset, geom_latents, is_train=True, train_stats=None, dataset_type='elasticity', use_conditions=False):
        """
        Args:
            base_dataset: Base dataset with coords and output fields
            geom_latents: Dict containing latents_p and latents_c
            is_train: If True, compute normalization stats. If False, use provided train_stats
            train_stats: Dict containing 'mean' and 'std' from training set normalization
            dataset_type: Either 'elasticity' or 'airfrans'
            use_conditions: Whether to use conditional features
        """
        self.base_dataset = base_dataset
        self.latents_p = geom_latents['latents_p']  # shape [N, L, latent_dim]
        self.latents_c = geom_latents['latents_c']  # shape [N, L, latent_dim]
        self.is_train = is_train
        self.dataset_type = dataset_type
        self.use_conditions = use_conditions
        
        if is_train:
            # Will compute normalization stats on first pass through dataset
            self.c_mean = None
            self.c_std = None
            self._compute_full_latents_stats()
        else:
            # Use provided training set statistics
            assert train_stats is not None, "Must provide train_stats for validation set"
            self.c_mean = train_stats['mean']
            self.c_std = train_stats['std']
    
    def _compute_full_latents_stats(self):
        """Compute normalization statistics on full concatenated latents_c."""
        # Get full concatenated latents for all samples
        full_latents = []
        for i in range(len(self)):
            sample = self.base_dataset[i]
            c = self.latents_c[i]  # [L, latent_dim]
            
            # Handle conditional features if requested
            if self.use_conditions and 'cond' in sample:
                c = self._append_conditions(c, sample['cond'])
            
            full_latents.append(c)
        
        # Stack all samples
        full_latents = jnp.stack(full_latents, axis=0)  # [N, L, full_latent_dim]
        
        # Compute statistics across batch and latent dimensions
        self.c_mean = jnp.mean(full_latents, axis=(0, 1), keepdims=False)  # [1, 1, full_latent_dim]
        self.c_std = jnp.std(full_latents, axis=(0, 1), keepdims=False)    # [1, 1, full_latent_dim]
        # Add small epsilon to avoid division by zero
        self.c_std = jnp.where(self.c_std == 0, 1e-8, self.c_std)
    
    def _append_conditions(self, c, cond):
        """Helper to append conditional features to latents."""
        cond = jnp.array(cond)
        if cond.ndim == 1:
            cond = cond[None, :]
        cond = jnp.tile(cond, [c.shape[0], 1])
        return jnp.concatenate([c, cond], axis=-1)

    def get_normalization_stats(self):
        """Return normalization statistics for use with validation set."""
        assert self.is_train, "Can only get stats from training dataset"
        return {
            'mean': self.c_mean,
            'std': self.c_std
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        coords = sample['input']     # shape [num_points, coord_dim]
        output_field = sample['output']  # shape [num_points, out_dim] -> PDE solution

        p = self.latents_p[idx]  # [L, latent_dim]
        c = self.latents_c[idx]  # [L, latent_dim]

        # Handle conditional features if requested
        if self.use_conditions and 'cond' in sample:
            c = self._append_conditions(c, sample['cond'])
        
        # Normalize full latents_c (including conditions if present)
        c = (c - self.c_mean) / self.c_std  # Now has mean 0, std 1

        # Basic fields that both datasets have
        result = {
            'input': coords,     # PDE domain coords
            'output': output_field, 
            'latents_p': p, 
            'latents_c': c,
        }
        
        # Add airfrans-specific fields if needed
        if self.dataset_type == 'airfrans':
            result.update({
                'normals': sample.get('normals', None),
                'is_airfoil': sample.get('is_airfoil', None),
            })
            
        return result

if __name__ == "__main__":
    data_directory = '/scratch/dmsm/gi.catalani/Elasticity/'
    dataset_name = 'elasticity'
    batch_size = 32
    if dataset_name=='Airfrans':
        target_field = 'all_outputs'
        train_data, train_names = af.dataset.load(root=data_directory, task='full', train=True)
        test_data, test_names = af.dataset.load(root=data_directory, task='full', train=False)

        # Create training dataset and compute normalization
        train_dataset = AirfransDataset(
            data_list=train_data,
            data_names=train_names,
            target_field=target_field,
            is_train=True,
            num_points=150000
        )

        # Create test dataset using the same normalization
        test_dataset = AirfransDataset(
            data_list=test_data,
            data_names=test_names,
            target_field=target_field,
            is_train=False,
            coef_norm=train_dataset.coef_norm,
            num_points=150000
        )
    elif dataset_name=='transonic':
        target_field = 'Pressure'
        dataset_object = JAXTransonicRAE(data_directory, target_field)
        dataset_object.create_splits()
        train_dataset = dataset_object.train_dataset
        val_dataset = dataset_object.val_dataset
        
    elif dataset_name=='elasticity':
       
        # Create datasets for both target fields
        for target_field in ['sigma', 'xy']:
            print(f"\nProcessing {target_field} target field:")
            dataset = ElasticityDataset(
                data_directory=data_directory,
                ntrain=1000,
                ntest=200,
                target_field=target_field
            )
            
            
            # Create dataloaders
            train_loader = JAXDataLoader(dataset.train_dataset, batch_size=32, shuffle=True)
            test_loader = JAXDataLoader(dataset.test_dataset, batch_size=32, shuffle=False)
            
            # Print shapes from first batch
            print(f"\nBatch shapes for {target_field}:")
            for batch in train_loader:
                print("Training batch:")
                print(f"pos shape: {batch['pos'].shape}")
                print(f"input shape: {batch['input'].shape}")
                print(f"output shape: {batch['output'].shape}")
                print(f"cond shape: {batch['cond'].shape}")
                break
            
            # Plot example data
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f'Target Field: {target_field}')
            
            if target_field == 'sigma':
                # Plot scalar field for sigma
                train_data = dataset.train_dataset[0]
                scatter = axs[0].scatter(
                    train_data['input'][:, 0], 
                    train_data['input'][:, 1], 
                    c=train_data['output'][:, 0], 
                    cmap='viridis'
                )
                axs[0].set_title('Training data')
                fig.colorbar(scatter, ax=axs[0])
                
                test_data = dataset.test_dataset[0]
                scatter = axs[1].scatter(
                    test_data['input'][:, 0], 
                    test_data['input'][:, 1], 
                    c=test_data['output'][:, 0], 
                    cmap='viridis'
                )
                axs[1].set_title('Test data')
                fig.colorbar(scatter, ax=axs[1])
            else:
                # Plot vector field for xy
                train_data = dataset.train_dataset[0]
                axs[0].quiver(
                    train_data['pos'][:, 0],
                    train_data['pos'][:, 1],
                    train_data['output'][:, 0],
                    train_data['output'][:, 1]
                )
                axs[0].set_title('Training data')
                
                test_data = dataset.test_dataset[0]
                axs[1].quiver(
                    test_data['pos'][:, 0],
                    test_data['pos'][:, 1],
                    test_data['output'][:, 0],
                    test_data['output'][:, 1]
                )
                axs[1].set_title('Test data')
            
            #plt.savefig(f'/home/dmsm/gi.catalani/Projects/ENF/experiments/elasticity_data_{target_field}.png')
            plt.close()
            
    