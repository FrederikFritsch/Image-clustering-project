#!/bin/env bash

# This first line is called a "shebang" and indicates what interpreter to use
# when running this file as an executable. This is needed to be able to submit
# a text file with `sbatch`

# The necessary flags to send to sbatch can either be included when calling
# sbatch or more conveniently it can be specified in the jobscript as follows

#SBATCH -A SNIC2021-7-164      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-10:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:4  # choosing no. GPUs and their type
#SBATCH -J FirstRun        # the jobname (not necessary)

module purge
module load SciPy-bundle/2020.11-fosscuda-2020b
module load matplotlib/3.3.3-fosscuda-2020b
module load OpenCV/4.5.1-fosscuda-2020b-contrib
module load scikit-learn/0.23.2-fosscuda-2020b

# The rest of this jobscript is handled as a usual bash script that will run
# on the primary node (in this case there is only one node) of the allocation
# Here you should make sure to run what you want to be run

DATA_PATH=/cephyr/NOBACKUP/groups/uu-it-gov/top20/frames/
#DATA_PATH=Image_Data/
RESULTS_FOLDER_NAME=DNN-VGG16-Big
IMAGE_SIZE_X=640
IMAGE_SIZE_Y=350
COLOR_FEATURES=1
ROI_FEATURES=0
EDGE_FEATURES=0
LBP_FEATURES=0
ORB_FEATURES=0

python3 TraditionalFeatureExtraction.py $DATA_PATH $RESULTS_FOLDER_NAME $IMAGE_SIZE_X $IMAGE_SIZE_Y $COLOR_FEATURES $ROI_FEATURES $EDGE_FEATURES $LBP_FEATURES $ORB_FEATURES
