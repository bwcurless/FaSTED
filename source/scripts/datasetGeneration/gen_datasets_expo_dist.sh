#!/bin/sh

#this corresponds to the dataset geenrator: dataset_gen_fixed_len_expo_dist.cpp
#lambda is 40.0
#creates datasets formatting for the GPU and SuperEGO

./dataset_gen_fixed_len_expo_dist 64 100000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 128 100000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 256 100000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 384 100000 /scratch/bc2497/datasets
