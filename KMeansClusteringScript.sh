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
module load SciPy-bundle/2020.11-fosscuda-2020b
module load matplotlib/3.3.3-fosscuda-2020b
module load OpenCV/4.5.1-fosscuda-2020b-contrib
module load scikit-learn/0.23.2-fosscuda-2020b

# The rest of this jobscript is handled as a usual bash script that will run
# on the primary node (in this case there is only one node) of the allocation
# Here you should make sure to run what you want to be run
CSV_PATH=TraditionalKmeans/TraditionalKmeans.csv
CSV_FOLDER_NAME=TraditionalKmeans
NORM_METHOD=Normalize           # Normalize/Standardize/MinMax
PCA=0.9                       #range (0,1) or nr_components
MIN_CLUSTERS=10
MAX_CLUSTERS=40
#TEST_PATH=/cephyr/NOBACKUP/groups/uu-it-gov/top20/frames/THKgFtr7J2w/
python3 KMeansClustering.py $CSV_PATH $CSV_FOLDER_NAME $NORM_METHOD $PCA $MIN_CLUSTERS $MAX_CLUSTERS
