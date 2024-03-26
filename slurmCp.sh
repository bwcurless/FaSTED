#!/bin/bash
# Copy over relevant files to slurm

scp debug/main wind:euclid/debug/
scp release/main wind:euclid/release/
scp ./Slurm_Debug_A100.sh wind:euclid/debug
