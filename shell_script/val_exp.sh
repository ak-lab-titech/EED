#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -j y
#$ -l h_rt=10:00:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

pwd

echo "LLaVA"

#cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval_exp --area-reward-type coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval_exp --area-reward-type novelty
CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval_exp --area-reward-type smooth-coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval_exp --area-reward-type curiosity
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval_exp --area-reward-type reconstruction

echo "LLaVA"