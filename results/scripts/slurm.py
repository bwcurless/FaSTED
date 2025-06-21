import os
from pathlib import Path


def get_node_tempdir() -> Path:
    """
    File accesses are slow on NFS, so request a temporary location on
    the node's local hard drive to store files to.
    """
    tmpdir = os.getenv("SLURM_TMPDIR")
    if tmpdir:
        return Path(tmpdir)
    else:
        raise Exception("Failed to get Slurm temporary directory.")


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
