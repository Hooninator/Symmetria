#!/usr/bin/bash
#SBATCH --nodes 16
#SBATCH --tasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -A m4646_g
#SBATCH -t 1:00:00


export NVSHMEM_SYMMETRIC_SIZE=2147483648 #Needed for 2D test

# 1D
#./driver.sh --action run_spsyrk --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 1d  --matname celegans --nodes 1
#./driver.sh --action run_spsyrk --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 1d  --matname celegans --nodes 4
./driver.sh --action run_spsyrk --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 1d  --matname celegans --nodes 16


# 2D
#./driver.sh --action run_spsyrk --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 2d  --matname celegans --nodes 1
#./driver.sh --action run_spsyrk --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 2d  --matname celegans --nodes 4
#./driver.sh --action run_spsyrk --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 2d  --matname celegans --nodes 16


#CombBLAS
#./driver.sh --action run_combblas --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 2d  --matname celegans --nodes 1
#./driver.sh --action run_combblas --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 2d  --matname celegans --nodes 4
./driver.sh --action run_combblas --ntrials 1 --rows 670567 --cols 1921940 --nnz 45811696 --type 2d  --matname celegans --nodes 16
