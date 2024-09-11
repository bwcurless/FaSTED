#!/bin/bash
# Script to compile .cu files and run sbatch
source ~/helpers/slurm_funcs.sh

# Exit if any part fails
set -e

# Input Arguments
target=$1
jobName=$2

# Add a dash on if we are customizing the filename
if [[ -n $jobName ]]; then
	jobPrefix=$jobName-
fi

outputFile="MMAPTXTest"

# Determine what target to build and run on
case $target in
	a100 | *)
		echo "Running on a100"
		CC=80
		cudaModule=12.3.2
		gpu=a100
		;;

	v100)
		echo "Running on v100"
		CC=70
		cudaModule=12.3.2
		gpu=v100
		;;

	p100)
		echo "Running on p100"
		CC=60
		cudaModule=12.3.2
		gpu=p100
		;;

	k80)
		echo "Running on k80"
		CC=37
		cudaModule=11.7 # k80's only run on this
		gpu=k80
		;;
esac

# Do a test build locally to make sure there aren't errors before waiting in queue
echo "Building executable to $outputFile"
module load "cuda/$cudaModule"
make clean
make release

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

#SBATCH --time=00:10:00
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C $gpu #GPU Model: k80, p100, v100, a100
#SBATCH --account=gowanlock_condo
#SBATCH --partition=gowanlock

# Code will not compile if we don't load the module
module load "cuda/$cudaModule"

# Can do arithmetic interpolation inside of $(( )). Need to escape properly
make

#srun ./release/main
#compute-sanitizer --tool=memcheck ./release/main
# -f overwrite profile if it exists
# --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
srun ncu -f -o "MMAPTXTest_profile_%i" --import-source yes --source-folder . --clock-control=none --set full ./release/main
#srun nsys profile ./main


echo "----------------- JOB FINISHED -------------"

SHELL
)


waitForJobComplete "$jobid"
printFile "$outputPath$jobPrefix$outputFile-$jobid.out"
