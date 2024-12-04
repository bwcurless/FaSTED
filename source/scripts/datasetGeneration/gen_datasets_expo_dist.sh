#!/bin/sh

#this corresponds to the dataset geenrator: dataset_gen_fixed_len_expo_dist.cpp
#lambda is 40.0
#creates datasets formatting for the GPU and SuperEGO

./dataset_gen_fixed_len_expo_dist 16 2000000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 32 2000000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 64 2000000 /scratch/bc2497/datasets

./dataset_gen_fixed_len_expo_dist 16 10000000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 32 10000000 /scratch/bc2497/datasets
./dataset_gen_fixed_len_expo_dist 64 10000000 /scratch/bc2497/datasets
