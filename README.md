# NormWear

This top-level README covers setup, downloads, and HPC usage for the repository root to run **zero-shot** evaluations only.

## Installation

### Using uv

```sh
# Creating the virtual environment
uv venv .venv
# Activating the virtual environment
source .venv/bin/activate
# Installing dependencies
uv sync
```

### Using pip

```sh
# Creating the virtual environment
python -m venv .venv
# Activating the virtual environment
source .venv/bin/activate
# Upgrading pip [Optional]
python -m pip install --upgrade pip
# Installing dependencies
pip install -r requirements.txt
```


## Download model weights

The pretrained NormWear  and the zero-shot MSiTF fusion checkpoints are available from the [GitHub release of the original repo](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha). After downloading the MSiTF checkpoint make sure to update `MSITF_CKPT_PATH` in the `.env` file to point at the checkpoint file.


## Download datasets

### Downstream datasets

The processed downstream datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing). After extraction, place the dataset folders under `NormWear/data/`.

### Pretraining datasets

The pretraining dataset archive can be downloaded from [Google Drive](https://drive.google.com/file/d/1WBlyweezkYm16PR3UFrO85XrZCKjqWmP/view?usp=sharing).

## TinyLlama path in `.env`

The zero-shot code loads  [muzammil-eds/tinyllama-2.5T-Clinical-v2](https://huggingface.co/muzammil-eds/tinyllama-2.5T-Clinical-v2) from HuggingFace by default. 
In case, the HPC does not allow requests to HuggingFace you can download the model then set the `TINYLLAMA_PATH` environment variable (in the `.env` file) to be the path of the downloaded model. 

```sh
TINYLLAMA_PATH=path/example
```

## HPC job script

`job_script_example.sh` is an example of a Slurm batch script for running NormWear on an HPC cluster. It does three things:

- Requests resources with `#SBATCH` options such as time limit, partition, account, GPUs, and log files.
- Loads the software stack with `module load`, then activates the virtual environment.
- Launches zero-shot inference with:

```sh
CUDA_VISIBLE_DEVICES=0 python3 -m NormWear.zero_shot.zero_shot_inference normwear --dataset wesad --times 3
```

Before submitting, update the module names, environment name, repository path, and Slurm account values to match your cluster. Then run:

```sh
sbatch job_script.sh
```

## Notes
 
This README file is about how to run Zero-shot evaluation on a HPC for a more detailed insight about the model please refer to [the original README](./NormWear/README.md)