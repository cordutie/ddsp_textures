#!/bin/bash
#SBATCH -J training_test_1
#SBATCH -p short
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20g
#SBATCH --time=2:00:00
#SBATCH -o %N.job%J.log_output.txt
#SBATCH -e %N.job%J.log_errors.txt

echo "### Loading modules..."
module --ignore-cache load CUDA
module load Python tqdm matplotlib
echo "### ...done."

echo "### Installing modules..."
python3 -m pip cache list
python3 -m pip install --upgrade pip
python3 -m pip install librosa
python3 -m pip install numpy
python3 -m pip install --upgrade numpy
python3 -m pip install torch
python3 -m pip install torchinfo
python3 -m pip install torchaudio
echo "### ...done."

echo "### HPC Job properties:"
echo "Number of Nodes Allocated     : $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated     : $SLURM_NTASKS"
echo "Number of Cores/Task Allocated: $SLURM_CPUS_PER_TASK"

echo "### Starting script 'main.py' ... $(date)"
python3 main.py
echo "###### Finished ###### $(date)"