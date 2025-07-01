# [ICML 2025] ELoRA: Low-Rank Adaptation for Equivariant GNNs

This repository contains the implementation of our ICML 2025 paper:

> **ELoRA: Low-Rank Adaptation for Equivariant GNNs**
> 
> Chen Wang, Siyu Hu, Guangming Tan, Weile Jia

The code is built upon the [e3nn](https://github.com/e3nn/e3nn) library and the [MACE](https://github.com/ACEsuit/mace) framework.

## Installation

The installation procedure is **based on the official setup of** [**MACE**](https://github.com/ACEsuit/mace), with modifications to support our ELoRA modules. We recommend using [**conda**](https://docs.conda.io/) to manage the environment.

Requirements:

- Python >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.12 **(training with float64 is not supported with PyTorch 2.1 but is supported with 2.2 and later.)**.

(for openMM, use Python = 3.9)

### Baseline Environment

This environment setup is used for the **full-parameter fine-tuning baselines** reported in our paper.

```bash
# Create a virtual environment and activate it
conda create --name mace_baseline
conda activate mace_baseline

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# (optional) Install MACE's dependencies from Conda as well
conda install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn

# Clone and install MACE (and all required packages)
pip install git+https://github.com/hyjwpk/ELoRA.git@MACE_baseline
```

For the Pytorch version, use the appropriate version for your CUDA version.

### ELoRA Environment

This environment setup is used for the **ELoRA fine-tuning** reported in our paper.

```bash
# Create a virtual environment and activate it
conda create --name mace_elora
conda activate mace_elora

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# (optional) Install MACE's dependencies from Conda as well
conda install numpy scipy matplotlib ase opt_einsum prettytable pandas

# Clone and install e3nn with ELoRA
pip install git+https://github.com/hyjwpk/ELoRA.git@main

# Clone and install MACE with ELoRA (and all required packages)
pip install git+https://github.com/hyjwpk/ELoRA.git@MACE_ELoRA
```

For the Pytorch version, use the appropriate version for your CUDA version.

## Training

**Baseline and ELoRA share the same command-line interface and training scripts.** To reproduce different results, simply **switch the active conda environment** (e.g., mace_baseline or mace_elora) before running the training commands.

For details on how to use **MACE**, please refer to the [official MACE README](https://github.com/hyjwpk/ELoRA/tree/MACE_baseline). Here, we provide the configurations used to reproduce the results reported in our ICML 2025 paper.

### Inorganic

```bash
mace_run_train \
    --name="MACE_model" \
    --train_file="./dataset/train.xyz" \
    --valid_file="./dataset/valid.xyz" \
    --E0s="average" \
    --foundation_model="2024-01-07-mace-128-L2_epoch-199.model" \
    --model="ScaleShiftMACE" \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=2 \
    --correlation=3 \
    --max_ell=3 \
    --r_max=6.0 \
    --max_L=2 \
    --num_channels=128 \
    --num_radial_basis=10 \
    --MLP_irreps="16x0e" \
    --scaling='rms_forces_scaling' \
    --loss="ef" \
    --energy_weight=1 \
    --forces_weight=1000 \
    --amsgrad \
    --lr=0.005 \
    --weight_decay=1e-8 \
    --batch_size=5 \
    --valid_batch_size=5 \
    --lr_factor=0.8 \
    --scheduler_patience=5 \
    --ema \
    --ema_decay=0.995 \
    --max_num_epochs=200 \
    --error_table="TotalRMSE" \
    --device=cuda \
    --seed=123 \
    --clip_grad=100 \
    --save_cpu 
```

### Organic

```bash
mace_run_train \
    --name="MACE_model" \
    --train_file="./dataset/train.xyz" \
    --valid_fraction=0.1 \
    --test_file="./dataset/test.xyz" \
    --E0s='average' \
    --foundation_model="MACE-OFF23_medium.model" \
    --model="MACE" \
    --loss="ef" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --r_max=5.0 \
    --lr=0.005 \
    --forces_weight=1000 \
    --energy_weight=1 \
    --weight_decay=1e-8 \
    --clip_grad=100 \
    --batch_size=5 \
    --valid_batch_size=5 \
    --max_num_epochs=500 \
    --scheduler_patience=5 \
    --ema \
    --ema_decay=0.995 \
    --error_table="TotalRMSE" \
    --default_dtype="float64"\
    --device=cuda \
    --seed=123 \
    --save_cpu 
```

You may adjust the hyperparameters or input files to suit your specific dataset or evaluation setting.

## Citation

The paper is available at https://openreview.net/forum?id=hcoxm3Vwgy&noteId=xEsvYXshHD

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wang2025elora,
  title={ELoRA: Low-Rank Adaptation for Equivariant GNNs},
  author={Wang, Chen and Hu, Siyu and Tan, Guangming and Jia, Weile},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

