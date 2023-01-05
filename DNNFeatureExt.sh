#!/bin/env bash

# This first line is called a "shebang" and indicates what interpreter to use
# when running this file as an executable. This is needed to be able to submit
# a text file with `sbatch`

# The necessary flags to send to sbatch can either be included when calling
# sbatch or more conveniently it can be specified in the jobscript as follows

#SBATCH -A SNIC2021-7-164      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-48:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:4  # choosing no. GPUs and their type
#SBATCH -J VGG16FeatureExctrationFrames        # the jobname (not necessary)

module purge
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
module load Pillow-SIMD/8.3.1-GCCcore-11.2.0
# The rest of this jobscript is handled as a usual bash script that will run
# on the primary node (in this case there is only one node) of the allocation
# Here you should make sure to run what you want to be run
DATA_PATH=/cephyr/NOBACKUP/groups/uu-it-gov/top20/frames
CSV_FOLDER_NAME=VGG16_Kmeans
# CURRENTLY SUPPORTED PRE-TRAINED MODELS ARE:
# - VGG16
# - Xception
MODEL_TYPE=VGG16
#TEST_PATH=/cephyr/NOBACKUP/groups/uu-it-gov/top20/frames/THKgFtr7J2w/
python3 NeuralNetworkFeatureExtraction.py $DATA_PATH $CSV_FOLDER_NAME $MODEL_TYPE



