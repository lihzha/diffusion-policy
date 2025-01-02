#!/bin/bash

#SBATCH --job-name=unet-robot    # Job name
#SBATCH --output=logs/%A.out   # Output file
#SBATCH --error=logs/%A.err    # Error file
#SBATCH --time=24:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:2            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=4        # Reduced CPU per task
#SBATCH --mem=30G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

# Module and environment setup
module load cudatoolkit/12.4


# Parameter configurations
CONFIGS=(
  "job_id=$SLURM_JOB_ID"
)

# GPU Check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# Function to find an open port
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

# Assign an unused port dynamically
export MASTER_PORT=$(find_free_port)


# Run script with selected configuration using torchrun
HYDRA_FULL_ERROR=1 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --standalone --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT scripts/run.py ${CONFIGS[0]}
