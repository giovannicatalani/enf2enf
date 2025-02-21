#!/bin/bash
#SBATCH --nodes=1                   #####Nb de noeuds
#SBATCH --ntasks-per-node=6        ##### Nb de coeurs/noeuds ici 24
#SBATCH --time=48:00:00             ##### temps demandé maximum de mon calcul
#SBATCH --mail-user=Giovanni.CATALANI2@isae-supaero.fr #### mon mail
#SBATCH --mail-type=FAIL,END                   #### mail quand c'est fini ou erreur
#SBATCH --job-name=enf_rae                  #### mon calcul
#SBATCH -o    slurm.%j.out                     #### sorti standard STDOUT
#SBATCH -e    slurm.%j.err                     #### sorti erreur STDERR
#SBATCH --partition=gpu02                       #### la partition
#SBATCH --gres=gpu:1
export {http,https,ftp,all}_proxy="http://proxy.isae.fr:3128"
####  loading python you want 2.7 or 3.7 ###
module purge
module load python/3.7
module list

##### Activate your environement ######
source deactivate
source activate py311_gdl

##### launch your program #####
#python elasticity.py
#python airfrans_full.py
python multi_element_airfoil.py
