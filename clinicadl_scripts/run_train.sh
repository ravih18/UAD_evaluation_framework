#!/bin/bash
#SBATCH --output=logs/slurm_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --job-name=train_VAE
#SBATCH --time=20:00:00

CAPS_DIR=/path/to/caps
PREPROCESSING_JSON=json_file
TSV_DIR=/path/to/tsv
MAPS_DIR=/path/to/maps
CONFIG_FILE=/config/config_VAE.toml

clinicadl train pythae $CAPS_DIR $PREPROCESSING_JSON $TSV_DIR $MAPS_DIR -c $CONFIG_FILE
