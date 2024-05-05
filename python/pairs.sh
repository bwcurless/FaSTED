#!/bin/bash
source ../slurm_funcs.sh

# Exit if any part fails
set -e

# Input Arguments
jobName=$1

# Add a dash on if we are customizing the filename
if [[ -n $jobName ]]; then
	jobPrefix=$jobName-
fi

outputFile="pairs"


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

#SBATCH --time=13:00:00
#SBATCH --mem=20000         #memory requested in MiB
#SBATCH -c 32 #Number of cpu cores

module load anaconda3
conda activate pairsEnv
srun --unbuffered python3 join.py ../../datasets/expo_16D_200000.txt 0.035


echo "----------------- JOB FINISHED -------------"

SHELL
)


waitForJobComplete "$jobid"
printFile "$outputPath$jobPrefix$outputFile-$jobid.out"
#scrollOutput "$outputPath$jobPrefix$outputFile-$jobid.out"
