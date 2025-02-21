import numpy as np
import torch
from torch_geometric.data import Data
import copy
from torch_geometric.data import Data, Dataset, DataLoader


class VariantSDFDataset(Dataset):
    def __init__(self, variants, is_train=True, coef_norm=None, num_points=None, transform=None, pre_transform=None):
        """
        variants: dictionary (from geom_db.npy) with keys mapping to dictionaries that contain:
            - "points": (M,2) numpy array of positions.
            - "sdf": (M,) numpy array of SDF values.
        is_train: if True, compute normalization parameters from the data.
        coef_norm: if is_train is False, provide normalization parameters as a dict.
        num_points: if provided, subsample each variant to this number of points.
        """
        super(VariantSDFDataset, self).__init__(None, transform, pre_transform)
        self.variants = variants
        self.num_points = num_points
        self.is_train = is_train
        
        if self.is_train:
            self.coef_norm = self.compute_norm_params(variants)
        else:
            if coef_norm is None:
                raise ValueError("coef_norm must be provided when is_train is False.")
            self.coef_norm = coef_norm
            
        self.data_list = self.process_variants(variants)

    def compute_norm_params(self, variants):
        """Compute normalization parameters from the training variants."""
        # Gather all positions and sdf values from all variants.
        all_positions = np.vstack([variants[k]['points'] for k in variants])
        all_sdf = np.hstack([variants[k]['sdf'] for k in variants])
        
        pos_norm = {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0)
        }
        sdf_mean = all_sdf.mean(axis=0)
        sdf_std = all_sdf.std(axis=0)
        coef_norm = {'pos_norm': pos_norm, 'mean': sdf_mean, 'std': sdf_std}
        print("Normalization parameters computed:")
        print(f"Position min: {pos_norm['min']}, max: {pos_norm['max']}")
        print(f"SDF mean: {sdf_mean}, std: {sdf_std}")
        return coef_norm

    def process_variants(self, variants):
        """Process each variant by normalizing positions and sdf, then creating a Data object."""
        data_list = []
        pos_min = self.coef_norm['pos_norm']['min']
        pos_max = self.coef_norm['pos_norm']['max']
        sdf_mean = self.coef_norm['mean']
        sdf_std = self.coef_norm['std']
        
        # Avoid division by zero in position normalization
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1.0  # Replace zeros with 1
        
        # Avoid division by zero in sdf normalization
        if sdf_std == 0:
            sdf_std = 1.0

        for key in variants:
            variant = variants[key]
            pts = variant['points']  # shape (M,2)
            sdf = variant['sdf']     # shape (M,)
            # Normalize positions to [-1,1] (using min-max scaling)
            pts_norm = 2 * (pts - pos_min) / pos_range - 1
            # Normalize sdf (zero mean, unit std)
            sdf_norm = (sdf - sdf_mean) / sdf_std
            # Remove any rows with NaN values in either pts_norm or sdf_norm
            mask = ~np.isnan(pts_norm).any(axis=1) & ~np.isnan(sdf_norm)
            pts_norm = pts_norm[mask]
            sdf_norm = sdf_norm[mask]
            # Subsample if necessary
            if self.num_points is not None and pts_norm.shape[0] > self.num_points:
                idx = np.random.choice(pts_norm.shape[0], self.num_points, replace=False)
                pts_norm = pts_norm[idx]
                sdf_norm = sdf_norm[idx]
            data = Data(
                pos=torch.tensor(pts_norm, dtype=torch.float),
                input=torch.tensor(pts_norm, dtype=torch.float),
                output=torch.tensor(sdf_norm, dtype=torch.float).unsqueeze(-1),
                is_airfoil=torch.zeros((pts_norm.shape[0],), dtype=torch.float)
            )
            data_list.append(data)
        return data_list


    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value
