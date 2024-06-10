#!/bin/bash
#SBATCH -J ddsp_textures_training
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --time=48:00:00
#SBATCH -o %N.job%J.log_output.txt
#SBATCH -e %N.job%J.log_errors.txt

echo "### Loading modules..."
module --ignore-cache load CUDA
module load Python tqdm
echo "### ...done."

echo "### Installing modules..."
python3 -m pip cache list
python3 -m pip install --upgrade pip
python3 -m pip install numpy
python3 -m pip install librosa
python3 -m pip install torch
python3 -m pip install torchinfo
echo "### ...done."

echo "### HPC Job properties:"
echo "Number of Nodes Allocated     : $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated     : $SLURM_NTASKS"
echo "Number of Cores/Task Allocated: $SLURM_CPUS_PER_TASK"

echo "### Starting trainer for ocean_short_mlp_stat' ... $(date)"
python3 ocean_short_mlp_stat_init.py
python3 ocean_short_mlp_stat_train.py
echo "###### Finished ###### $(date)"