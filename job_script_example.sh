#!/bin/bash

# BEGIN MANDATORY OPTIONS
#SBATCH --time=0-00:00:00       # Time limit DD-HH:MM:SS
#SBATCH --qos= CHANGE_ME            # priority/quality of service, please refer to the your HPC documentation for more details about available QoS options.
#SBATCH --account= CHANGE_ME      # Replace with your project/accout name
# END MANDATORY OPTIONS

# BEGIN INFORMATIONAL OPTIONS
#SBATCH --job-name= CHANGE_ME        # Name for your job
#SBATCH --comment= CHANGE_ME # Comment for your job
#SBATCH --output=%j.out      # Output file
#SBATCH --error=%j.err       # Error file
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail on start and end job
#SBATCH --mail-user= CHANGE_ME  # Email address where to recieve the notifications about your job
# END INFORMATIONAL OPTIONS

# BEGIN RESOURCES OPTIONS
#SBATCH --partition= CHANGE_ME      # select partition (queue) to run your job, please refer to the your HPC documentation for more details about available partitions. or run "sinfo" command to get the list of available partitions.
#SBATCH --ntasks= CHANGE_ME            # How many CPUs to use
#SBATCH --gres= CHANGE_ME          # How many GPUs to use (make sure that the partition you are using has GPU nodes))
# END RESOURCES OPTIONS


# BEGIN JOB

# Load the necessary modules 
module purge # Clear all loaded modules to avoid conflicts
module load cuda/12.8.0_570.86.10

cd "CHANGE_ME" # Change to the directory where your to the directory where your this script is located or where your project is located

# Activate your virtual environment that is already set up with the required dependencies for your project as shown in the README file
source .venv/bin/activate

# Run your Python script (see the README file for the exact command to run your script, make sure to replace the dataset name and number of runs with the appropriate values)
CUDA_VISIBLE_DEVICES=0 python3 -m NormWear.zero_shot.zero_shot_inference normwear --dataset wesad 
