#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=titanx
#SBATCH --job-name=trans_base_legacy
#SBATCH --output=trans_base_legacy.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00 # This is the max
#SBATCH --qos=long

# Load modules
module restore

# Run the job
srun python ../model/train.py -ngpus 2 -bsize 89 -fw transformer -out trans_base_legacy -layer_drop 0.2 -op adagrad -lr 0.1 --mode wikilegacy -nhl 4 -nel 4 -ndl 4 -lc True

