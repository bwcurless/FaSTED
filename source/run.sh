#!/bin/bash
#SBATCH --job-name=cuSimSearch  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/bc2497/cuSimSearch-%j.out
#SBATCH --error=/scratch/bc2497/cuSimSearch-%j.out

#SBATCH --time=01:00:00
#SBATCH --mem=128000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100

# Gowanlock Partition
#SBATCH --account=gowanlock_condo
#SBATCH --partition=gowanlock

# Main partition
###SBATCH --account=gowanlock

#SBATCH --exclusive=user

set -e

# Code will not compile if we don't load the module
module load cuda

# Can do arithmetic interpolation inside of $(( )). Need to escape properly
make clean
make

#srun nvidia-smi
#srun ./release/main "/scratch/bc2497/datasets/tiny5m_unscaled.txt" 0.2
#srun ./release/main "/home/bc2497/datasets/expo_16D_2000000.txt" 0.035
#srun ./release/main -e  100000 4096 0.1
#compute-sanitizer --tool=memcheck ./release/main "/scratch/bc2497/datasets/bigcross.txt" 0.03
#compute-sanitizer --tool=memcheck --launch-timeout=4000 ./release/main -e 1000000 4096 0.001
#compute-sanitizer --tool=racecheck ./release/main "$HOME/datasets/expo_16D_200000.txt" 0.001
# -f overwrite profile if it exists
# --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
srun ncu -f -o "MMAPTXTest_profile_expo_100000_128_%i" --import-source yes --source-folder . --clock-control=none --set full ./release/main /scratch/bc2497/datasets/tedjoin_expo_data/dataset_fixed_len_pts_expo_NDIM_128_pts_100000.txt 0.2873
srun ncu -f -o "MMAPTXTest_profile_expo_100000_4096_%i" --import-source yes --source-folder . --clock-control=none --set full ./release/main -e 100000 4096 0.2873
#srun nsys profile ./main


echo "----------------- JOB FINISHED -------------"
