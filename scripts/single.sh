#!/bin/bash

#SBATCH --job-name=unet-robot    # Job name
#SBATCH --output=logs/%A.out   # Output file
#SBATCH --error=logs/%A.err    # Error file
#SBATCH --time=72:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=8         # Reduced CPU per task
#SBATCH --mem=20G                    # Memory per node
#SBATCH --partition=all              # Or specify GPU partition if needed

# Module and environment setup
module load cudatoolkit/12.4

# Parameter configurations
CONFIGS=(
  # "model.network._target_=model.diffusion.unet.VisionUnet1D"
  # "model.network.backbone.cfg.num_patch_embs=1 model.network._target_=model.diffusion.unet.VisionUnet1D"
  "model.network._target_=model.diffusion.unet.VisionUnet1D train_dataset.use_delta_actions=False job_id=$SLURM_JOB_ID"
  # "model.network._target_=model.diffusion.mlp_diffusion.VisionDiffusionMLP model.network.mlp_dims=[1024,2048,1024] job_id=$SLURM_JOB_ID --config-name=pre_diffusion_mlp_img.yaml"
  # "model.network._target_=model.diffusion.mlp_diffusion.VisionDiffusionMLP model.network.mlp_dims=[1024,2048,1024] job_id=$SLURM_JOB_ID train.batch_size=128 --config-name=pre_diffusion_mlp_img.yaml "
  # "model.network._target_=model.diffusion.unet.VisionUnet1D obs_dim=8 train_dataset_path=/n/fs/robot-data/guided-data-collection/guided_dc/diffusion/data/traj_data/processed_dataset_no_vel.npz job_id=$SLURM_JOB_ID"
  # "model.network._target_=model.diffusion.unet.VisionUnet1D obs_dim=8 train_dataset_path=/n/fs/robot-data/guided-data-collection/guided_dc/diffusion/data/traj_data/processed_dataset_no_vel.npz use_delta_actions=False job_id=$SLURM_JOB_ID"
  # "model.network._target_=model.diffusion.mlp_diffusion.VisionDiffusionMLP model.network.mlp_dims=[1024,1024,1024] --config-name=pre_diffusion_mlp_img.yaml obs_dim=8 train_dataset_path=/n/fs/robot-data/guided-data-collection/guided_dc/diffusion/data/traj_data/processed_dataset_no_vel.npz job_id=$SLURM_JOB_ID"
)
# Select configuration
# OVERRIDES=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# GPU Check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run script with selected configuration
HYDRA_FULL_ERROR=1 python guided_dc/scripts/run.py ${CONFIGS[0]}
