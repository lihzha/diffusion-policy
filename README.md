# Guided Data Collection

## Installation 

1. Clone the repository
```console
git clone git@github.com:irom-lab/guided-data-collection.git
cd guided-data-collection
git submodule update --init --recursive
```

2. Install core dependencies and the maniskill submodule with a conda environment on a Linux machine with a Nvidia GPU.
```console
source install.sh
```

3. Set up path variables.
```console
./guided_dc/diffusion/scripts/set_path.sh
```

## Usage for training diffusion policy

### Single GPU

Modify task cfg file `guided_dc/cfg/real/diffusion_unet_vit.yaml` or `guided_dc/cfg/real/diffusion_unet_resnet.yaml` to try different training configurations, e.g., larger ViT.

```console
cd guided_dc
sbatch scripts/single.sh
```

### Multi GPU

Note that in `guided_dc/diffusion/multi_gpu.sh`, nproc_per_node should match the number of requested GPUs. The multi gpu version uses DistributedDataParallel to leverage multiple GPUs for training: it creates multiple processes, each with one GPU, and copy the same model to all processes. During each forward and backward, it splits the original batch into mini-batches and pass each mini-batch to a process. Losses and gradients are computed parallely for all mini-batches, and gradients are then averaged to back propogate models on all processes simutaneously.  

```console
cd guided_dc
sbatch scripts/single.sh
```

## Usage for simulation

### Running a task

You can define task configurations (e.g., initialization, randomization) in `cfg/config.yaml`. You can also override the default arguments from the command line.

```console
python guided_dc/scripts/random_action.py env_id=PickAndPlace-v1 num_envs=5
```

### Defining a task

All tasks are defined under the folder `envs/tasks`. To define a custom task, you can follow the template in `envs/tasks/push_objects.py`.


