#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=tuned_subtok
#SBATCH --output=tuned_subtok.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/eval.py -ngpus 1 -bsize 256 -fw transformer -out tuned_subtok -layer_drop 0.0 -op adagrad -lr 0.01 --mode trans -dim 300 -nh 4 -nhl 4 -nel 4 -ndl 4 -lc True -eval_freq 0 --fetch_mode tf_example_dataset --tune_style 0.0:1.0:0.0:0.0 --tune_mode plus:decoder:cond --subword_vocab_size 30000 --dmode alter

