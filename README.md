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
In case you want to use the model from HuggingFace, First download it localy
```sh
hf download mosaic-laboratory/normwear
```
**Note**:
* To use the weights from the github checkpoints use [zero_shot_inference.py](NormWear/zero_shot/zero_shot_inference.py) and to use the hugging face model use [zero_shot_inference_HF.py](NormWear/zero_shot/zero_shot_inference_HF.py); Both the files are the same the only difference is that one use the NormWearZeroShot class and the other one use NormWearZeroShotHF (both classes are provided in [msitf_fusio.py](NormWear/zero_shot/msitf_fusion.py))
* In case you want to costumize the directory of the model use this commend: 
```sh
hf download mosaic-laboratory/normwear --local-dir YOUR_DIRECTORY
```
and set the variable in the [.env](.env) file
```sh
NORMWEAR_PATH=YOUR_DIRECTORY
```
## Download datasets

### Downstream datasets

The processed downstream datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1Mojf_iby8FnUydogwUE-b321fB1V6SGK/view?usp=sharing). After extraction, place the dataset folders under `NormWear/data/`.

### Pretraining datasets

The pretraining dataset archive can be downloaded from [Google Drive](https://drive.google.com/file/d/1WBlyweezkYm16PR3UFrO85XrZCKjqWmP/view?usp=sharing).

## TinyLlama path in `.env`

The zero-shot code loads  [muzammil-eds/tinyllama-2.5T-Clinical-v2](https://huggingface.co/muzammil-eds/tinyllama-2.5T-Clinical-v2) localy. 
You can download the model localy by using: 
```sh
hf download muzammil-eds/tinyllama-2.5T-Clinical-v2
```
**Note**: In case you want to costumize the directory of the model use this commend:
```sh
hf download muzammil-eds/tinyllama-2.5T-Clinical-v2 --local-dir YOUR_DIRECTORY
```
and set the variable in the [.env](.env) file
```sh
TINYLLAMA_PATH=YOUR_DIRECTORY
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