import os


def running_on_slurm():
    """
    Checks if the script is being run in a slurm job.
    """
    on_slurm = "SLURM_JOB_ID" in os.environ
    if on_slurm:
        print("Running on slurm")
    else:
        print("Not running on slurm")
    return on_slurm
