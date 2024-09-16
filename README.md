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

## Usage

### Running a task

You can define task configurations (e.g., initialization, randomization) in `cfg/config.yaml`. You can also override the default arguments from the command line.

```console
python guided_dc/scripts/random_action.py env_id=PickAndPlace-v1 num_envs=5
```

### Defining a task

All tasks are defined under the folder `envs/tasks`. To define a custom task, you can follow the template in `envs/tasks/push_objects.py`.