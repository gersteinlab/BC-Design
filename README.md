# BC-Design: A Biochemistry-Aware Framework for High-Precision Inverse Protein Folding
<p align="left">
<a href="https://github.com/gersteinlab/BC-Design/blob/public-release/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<a href="https://github.com/gersteinlab/BC-Design/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/gersteinlab/BC-Design?color=%23FF9600" /></a>
<a href="https://img.shields.io/github/stars/gersteinlab/BC-Design/" alt="arXiv">
    <img src="https://img.shields.io/github/stars/gersteinlab/BC-Design" /></a>
</p>

<!-- [📘Documentation](https://openstl.readthedocs.io/en/latest/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](docs/en/model_zoos/video_benchmarks.md) |
[🆕News](docs/en/changelog.md) -->

This repository contains the implementation code for the paper:

[**BC-Design: A Biochemistry-Aware Framework for High-Precision Inverse Protein Folding**]

Xiangru Tang<sup>†</sup>, Xinwu Ye</sup>†</sup>, Fang Wu</sup>†</sup>, Daniel Shao, Yin Fang, Siming Chen, Dong Xu, and Mark Gerstein.

<sup>†</sup> Equal contribution

![image](./assets/BC-Design.png)

## Introduction

Inverse protein folding aims to design amino acid sequences that form specific 3D structures, which is crucial for protein engineering and drug development. Traditional approaches often neglect vital biochemical characteristics that impact protein function. BC-Design introduces a new approach that combines structural data and biochemical attributes, using a dual-encoder architecture for enhanced accuracy. This framework, which surpasses current methods in sequence recovery and structural precision, demonstrates strong generalization and performs well with complex protein features.

**Key Features of BC-Design:**
  - Integrates structural and biochemical features for protein design.
  - Uses a dual-encoder system with a Structure Encoder for spatial relationships and a BC-Encoder for biochemical features.
  - A BC-Fusion module enables cross-modal feature interaction, enhancing alignment of structural and biochemical data.
  - Outperforms traditional methods with high sequence recovery (88.37%) and low perplexity (1.47) on the CATH 4.2 benchmark.
  - Exhibits robust generalization across diverse protein sizes, complexity levels, and structural classes.

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview

<details open>
<summary>Code Structures</summary>

![image](./assets/BC-Design-overview.png)

- `src/datasets` contains datasets, featurizer, and utils
- `src/interface` contains customized Pytorch-lightning data modules and modules.
- `src/models/` contains the main BC-Design model architecture.
- `src/tools` contains some script files of some tools.
- `train` contains the training and inference script.

</details>

## News and Updates

- [🚀 2024-10-30] The official code is released.


## ⚙️ Installation

This section guides you through setting up the necessary environment and dependencies to run BC-Design.

### Step 1: Prerequisites - CUDA and GCC

Before creating the Conda environment, please ensure your system meets the following requirements:

1.  **CUDA Version:** This project requires **NVIDIA driver support for CUDA 12.1.1**.
    * You can check your NVIDIA driver version by running `nvidia-smi`. Ensure it's compatible with CUDA 12.1.1. The Conda environment will install the specific CUDA toolkit, but your system's driver must be compatible.
2.  **GCC Compiler:** A C/C++ compiler is needed, specifically **GCC version 12.2.0** or a compatible version.
    * **Linux:** You can typically install GCC using your system's package manager. For example, on Debian/Ubuntu-based systems, you might use:
        ```shell
        sudo apt update
        sudo apt install gcc-12 g++-12
        ```
        On other distributions, use the appropriate package manager (e.g., `yum`, `dnf`). You may need to configure your system to use this specific version if multiple GCC versions are installed.
    * **HPC Environments:** If you are using a High-Performance Computing (HPC) cluster, GCC is often managed via environment modules. You might load it using a command like:
        ```shell
        module load gcc/12.2.0
        ```
        (The exact command may vary based on your HPC's module system.)
    * **Other Systems (macOS, Windows via WSL2):** Ensure you have a compatible C/C++ compiler. For macOS, Xcode Command Line Tools provide Clang, which is often compatible. For Windows, WSL2 with a Linux distribution is recommended.

### Step 2: Create Conda Environment

This project has provided an environment setting file for **Miniconda3**. Users can easily reproduce the Python environment by following these commands:

```shell
git clone https://github.com/gersteinlab/BC-Design.git
cd BC-Design
conda env create -f environment.yml -n [your-env-name]
conda activate [your-env-name]
````

Replace `[your-env-name]` with your preferred name for the Conda environment (e.g., `bcdn`).

### Step 3: Download Data and Model Checkpoint

To train the model or run inference with the pre-trained checkpoint, you need to download the necessary data and the model weights.

1.  Navigate to the OSF project page: [https://osf.io/pwbhg/files/osfstorage](https://osf.io/pwbhg/files/osfstorage)
2.  Download the following files into the `BC-Design` folder (the main directory cloned from GitHub):
      * `data.zip` (contains data for training and inference)
      * `BC-Design.ckpt` (the pre-trained model checkpoint for inference)
3.  Once downloaded, unzip the data file:
    ```shell
    unzip data.zip
    ```
    This should create a `data/` directory inside your `BC-Design` folder.

As an alternative, you can also run the following commands:
```shell
wget https://osf.io/download/683dcbe71618e6327085b39f/ -O BC-Design.ckpt
wget https://osf.io/download/683dd27930c7903aaf85b1f7/ -O data.zip
unzip data.zip
````

After completing these steps, your environment should be ready, and you'll have the necessary data and model checkpoint to proceed with using BC-Design.


## Getting Started

<!-- **Obtaining Dataset**

The processed datasets could be found in the [releases](https://github.com/A4Bio/ProteinInvBench/releases/tag/dataset_release).  -->

**Model Training**

```shell
python train/main_fused.py
```

**Model Inference**
```shell
python train/main_eval.py --dataset CATH4.2
python train/main_eval.py --dataset TS50
python train/main_eval.py --dataset TS500
python train/main_eval.py --dataset AFDB2000
```

<p align="right">(<a href="#top">back to top</a>)</p>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

<!-- ## Citation -->


## Contribution and Contact

For adding new features, looking for helps, or reporting bugs associated with `BC-Design`, please open a [GitHub issue](https://github.com/gersteinlab/BC-Design/issues) and [pull request](https://github.com/gersteinlab/BC-Design/pulls) with the tag "new features", "help wanted", or "enhancement". Please ensure that all pull requests meet the requirements outlined in our [contribution guidelines](https://github.com/gersteinlab/BC-Design/blob/public-release/CONTRIBUTING.md). Following these guidelines helps streamline the review process and maintain code quality across the project.
Feel free to contact us through email if you have any questions.


<p align="right">(<a href="#top">back to top</a>)</p>
