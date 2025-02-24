#!/bin/bash
#SBATCH -J ddsp_textures_retraining
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64g
#SBATCH --time=72:00:00
#SBATCH -o %N.job%J.log_output.txt
#SBATCH -e %N.job%J.log_errors.txt

# Retrieve the configuration file from the environment variable
MODEL_FOLDER=$1

if [ -z "$MODEL_FOLDER" ]; then
  echo "Error: No model folder file provided."
  exit 1
fi

echo "### Loading modules..."
module --ignore-cache load CUDA
module load Python tqdm
echo "### ...done."

echo "### Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "### ...done."

echo "### HPC Job properties:"
echo "Number of Nodes Allocated     : $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated     : $SLURM_NTASKS"
echo "Number of Cores/Task Allocated: $SLURM_CPUS_PER_TASK"

echo "### Starting re-trainer for model in $MODEL_FOLDER ... $(date)"
python3 main.py retrain $MODEL_FOLDER
echo "###### Finished ###### $(date)"
