# Equivariant Neural Field Networks for PDE Solutions

This repository contains the implementation of enf2enf, a neural operator approach to solve steady state PDEs on general geometries.

## Architecture Overview

Our architecture leverages equivariant neural fields to learn PDE solutions across different domains. The network processes geometric features and produces accurate field predictions while maintaining equivariance properties.

![Architecture Overview](figures/architecture_skectch_v4-1.png)

## Experiments

### Elasticity Problem

The model learns deformation fields in elastic materials,

![Stress Plots](figures/results_elasticity-1.png)

```bash
python elasticity.py
```

### Airfrans Dataset

For airfoil simulations, we use the AirFRANS dataset to predict flow fields around airfoils at different angles of attack and flow conditions. To run experiments:
![Pressure Distributions](figures/comparison_sample_81_-0.7650_92.7220-1.png)

```bash
python airfrans_full.py
```

## Data
Data for Airfrans dataset can be found by downloading the airfrans package.
Data for elasticity dataset can be found at https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8 .