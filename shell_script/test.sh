#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -j y
#$ -l h_rt=00:05:00
#$ -o output/o.$JOB_ID
#$ -p -3

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

pip freeze > requirements.txt

#python speaker_policy.py
#python search_similar_description.py
#python make_hes_log.py
#python test_llava/test_llava.py
#python test_llava/test_llava2.py
#python test_llava/test_llava_next.py
#python human_metrics.py
#cd SentenceBert_FineTuning
#python finetuning_sbert.py
#python finetuning_sbert2.py
#python finetuning_sbert3.py
#python finetuning_sbert4.py
#python finetuning_sbert5.py
#python ShowingFinetuninng.py


echo "Finish !!!"