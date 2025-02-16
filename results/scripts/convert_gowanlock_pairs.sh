#!/bin/bash
#SBATCH --job-name=Accuracy_Python_Script  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/bc2497/Accuracy_%j.out
#SBATCH --error=/scratch/bc2497/Accuracy_%j.out

#SBATCH --time=1:00:00
#SBATCH --mem=30G         #memory requested in MiB
#SBATCH --cpus-per-task=4 #resource requirement

# Main partition
#SBATCH --account=gowanlock

# Abort if any command fails
set -e

module load anaconda3

conda activate myenv

python -u convert_gowanlock_pairs.py
#python -u compute_accuracy.py

echo "----------------- JOB FINISHED -------------"
