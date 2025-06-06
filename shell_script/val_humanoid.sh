#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=10:00:00
#$ -o output/o.$JOB_ID
#$ -p -3

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

pwd

#cd /gs/fs/tga-aklab/matsumoto/Main3
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

CUDA_LAUNCH_BLOCKING=1 python run_humanoid.py --run-type eval --area-reward-type coverage