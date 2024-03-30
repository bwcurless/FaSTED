#!/bin/bash
#SBATCH --job-name=A100_euclid_debug  #the name of your job

#change to your NAU ID below
#SBATCH --output=/scratch/bc2497/A100_euclid_debug.out #this is the file for stdout
#SBATCH --error=/scratch/bc2497/A100_euclid_debug.err #this is the file for stderr

#SBATCH --time=00:13:00		#Job timelimit is 3 minutes
#SBATCH --mem=20000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100

module load cuda
#compute-sanitizer --tool memcheck main expo_16D_262144.txt 0.001 42
./release/main ~/datasets/expo_16D_200000.txt 0.1 4
./release/main ~/datasets/expo_16D_200000.txt 0.1 42
./release/main ~/datasets/expo_16D_200000.txt 0.1 43
./release/main ~/datasets/expo_16D_200000.txt 0.1 44
