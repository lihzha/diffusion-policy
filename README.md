# Diffusion Policy for Robot Policy Learning

## Installation

1. Clone the repository
```console
git clone --recurse-submodules git@github.com:lihzha/diffusion-policy.git
```

2. Install core dependencies and the maniskill submodule with a conda environment on a Linux machine with a Nvidia GPU.
```console
cd guided-data-collection
conda create -n gdc python=3.9 -y
conda activate gdc
pip install -e .

cd guided_dc/maniskill
pip install -e .
cd ../..
```

3. Set up path variables.
```console
./scripts/set_path.sh
```

## Usage for training diffusion policy

### Single GPU

Modify task cfg file `guided_dc/cfg/real/diffusion_unet_vit.yaml` or `guided_dc/cfg/real/diffusion_unet_resnet.yaml` to try different training configurations, e.g., larger ViT.

```console
sbatch scripts/single.sh
```

### Multi GPU

Note that in `./scripts/multi_gpu.sh`, nproc_per_node should match the number of requested GPUs. The multi gpu version uses DistributedDataParallel to leverage multiple GPUs for training: it creates multiple processes, each with one GPU, and copy the same model to all processes. During each forward and backward, it splits the original batch into mini-batches and pass each mini-batch to a process. Losses and gradients are computed parallely for all mini-batches, and gradients are then averaged to back propogate models on all processes simutaneously.

```console
sbatch scripts/single.sh
```

## Usage for simulation

### Run simulation evaluation

```console
python scripts/simulation/eval_sim.py -j 1056700 -c 200
```

### Running a task

You can define task configurations (e.g., initialization, randomization) in `cfg/config.yaml`. You can also override the default arguments from the command line.

```console
python scripts/simulation/random_action.py env_id=PickAndPlace-v1 num_envs=5
```

### Defining a task

All tasks are defined under the folder `envs/tasks`. To define a custom task, you can follow the template in `envs/tasks/push_objects.py`.
