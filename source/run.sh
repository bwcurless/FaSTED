#!/bin/bash
# Script to compile .cu files and run sbatch

# You will need to copy these helper routines to this location to run this script.
source ~/helpers/helper_slurm_funcs.sh

# Exit if any part fails
set -e

# Input Arguments
jobName=$1
CC=80
gpu=a100

# Add a dash on if we are customizing the filename
if [[ -n $jobName ]]; then
jobPrefix=$jobName-
fi

outputFile="MMAPTXTest"

# Do a test build locally to make sure there aren't errors before waiting in queue
#echo "Building executable to $outputFile"
#module load cuda
#make clean
#make

# Define where outputs go
outputPath="/scratch/$USER/"
errorPath="$outputPath"

echo "Executing..."

jobid=$(sbatch --parsable <<SHELL
#!/bin/bash
#SBATCH --job-name=$jobPrefix$outputFile  #the name of your job

#change to your NAU ID below
#SBATCH --output=$outputPath$jobPrefix$outputFile-%j.out
#SBATCH --error=$errorPath$jobPrefix$outputFile-%j.out

#SBATCH --time=01:00:00
#SBATCH --mem=128000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C $gpu #GPU Model: k80, p100, v100, a100

# Gowanlock Partition
###SBATCH --account=gowanlock_condo
###SBATCH --partition=gowanlock

# Main partition
#SBATCH --account=gowanlock

# Output git information for traceability of what was run
echo "Commit Hash: $(git rev-parse HEAD)"
echo "Git Diff"
git --no-pager diff
echo "Git Diff Staged"
git --no-pager diff --staged

set -e

# Code will not compile if we don't load the module
module load cuda

# Can do arithmetic interpolation inside of $(( )). Need to escape properly
make clean
make

#srun nvidia-smi
#srun ./release/main "/scratch/bc2497/datasets/tiny5m_unscaled.txt" 0.2
#srun ./release/main "/home/bc2497/datasets/expo_16D_2000000.txt" 0.035
#srun ./release/main -e  100000 4096 0.003
#compute-sanitizer --tool=memcheck ./release/main "/scratch/bc2497/datasets/bigcross.txt" 0.03
#compute-sanitizer --tool=memcheck --launch-timeout=4000 ./release/main -e 1000000 4096 0.001
#compute-sanitizer --tool=racecheck ./release/main "$HOME/datasets/expo_16D_200000.txt" 0.001
# -f overwrite profile if it exists
# --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
srun ncu --list-sets
srun ncu --list-sections
#srun ncu -f -o "MMAPTXTest_profile_%i" --import-source yes --source-folder . --clock-control=none --set pmsampling_warpstates ./release/main -e 100000 2048 0.003
#srun nsys profile ./main


echo "----------------- JOB FINISHED -------------"

SHELL
)


#waitForJobComplete "$jobid"
#printFile "$outputPath$jobPrefix$outputFile-$jobid.out"
