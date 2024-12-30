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

outputFile="MMAPTXPythonScript"

# Do a test build locally to make sure there aren't errors before waiting in queue
echo "Building executable to $outputFile"
#module load cuda
#make clean
#make shared

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

#SBATCH --time=11:10:00
#SBATCH --mem=50000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C $gpu #GPU Model: k80, p100, v100, a100

# Gowanlock Partition
##SBATCH --account=gowanlock_condo
##SBATCH --partition=gowanlock

# Main partition
#SBATCH --account=gowanlock

# Abort if any command fails
set -e

# Code will not compile if we don't load the module
module load cuda
module load python/3.10.8
module load py-pip
pip install numpy

cd "../../source"
make shared

# Navigate back
cd -
python -um run_experiments
#srun ./release/main "$HOME/datasets/expo_16D_200000.txt" 0.03
#compute-sanitizer --tool=memcheck ./release/main "$HOME/datasets/expo_16D_200000.txt" 0.03
#compute-sanitizer --tool=racecheck ./release/main "$HOME/datasets/expo_16D_200000.txt" 0.001
# -f overwrite profile if it exists
# --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
#srun ncu -f -o "MMAPTXTest_profile_%i" --import-source yes --source-folder . --clock-control=none --set full ./release/main "$HOME/datasets/expo_16D_200000.txt" 0.03
#srun nsys profile ./main


echo "----------------- JOB FINISHED -------------"

SHELL
)


#waitForJobComplete "$jobid"
#printFile "$outputPath$jobPrefix$outputFile-$jobid.out"
