## Usage

### Pre-process data

```console
python script/dataset/process_real_dataset.py
```
<!-- The data path follows `${DPPO_DATA_DIR}/<benchmark>/<task>/train.npz`, e.g., `${DPPO_DATA_DIR}/gym/hopper-medium-v2/train.npz`. -->

### Run pre-training with data
Configs can be found at `?`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
```console
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2
```
