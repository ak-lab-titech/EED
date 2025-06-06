#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -j y
#$ -l h_rt=00:30:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

pwd

#cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

#pip install segment-anything

#python kachaka_picture.py
python test_llava/test_llava3.py