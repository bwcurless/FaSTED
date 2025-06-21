#!/bin/bash
#SBATCH --job-name=compute_accuracy  #the name of your job

#SBATCH --output=/scratch/bc2497/compute_accuracy-%j.out
#SBATCH --error=/scratch/bc2497/compute_accuracy-%j.out

#SBATCH --time=10:00:00
#SBATCH --mem=100000         #memory requested in MiB

# Main partition
#SBATCH --account=gowanlock

# Abort if any command fails
set -e

# Code will not compile if we don't load the module
module load python/3.10.8
module load py-pip
pip install numpy

python -m compute_accuracy_real_datasets


echo "----------------- JOB FINISHED -------------"
