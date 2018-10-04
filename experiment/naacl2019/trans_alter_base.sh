#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=trans_alter_base
#SBATCH --output=trans_alter_base.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/train.py -dmode alter -ngpus 1 -bsize 32 -fw transformer -out alter_base -layer_drop 0.2 -op adagrad -lr 0.1 --mode trans -dim 512 -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset
