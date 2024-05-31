#!/bin/bash
#SBATCH -J create_env_ddsp_textures
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=1:00:00
#SBATCH -o %N.job%J.log_output.txt
#SBATCH -e %N.job%J.log_errors.txt

echo "### Creating conda environment 'ddsp_textures'..."
conda create -y -n ddsp_textures python=3.8

echo "### Activating conda environment 'ddsp_textures'..."
source activate ddsp_textures

echo "### Installing packages..."
conda install -y -c pytorch torch torchaudio librosa numpy scipy
pip install tqdm matplotlib

echo "### Verifying installed packages..."
conda list

echo "### HPC Job properties:"
echo "Number of Nodes Allocated     : $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated     : $SLURM_NTASKS"
echo "Number of Cores/Task Allocated: $SLURM_CPUS_PER_TASK"

echo "###### Environment 'ddsp_textures' creation finished ######"
