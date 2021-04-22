#!/bin/bash

# set a job name
#SBATCH --job-name=deit_miniIN
#################

# a file for job output, you can check job progress
#SBATCH --output=/gpfs/u/home/LLLD/LLLDashr/log_cdfsl_fewshot/deit_miniIN/slurm_out_logs/%j.out
#################

# a file for errors
#SBATCH --error=/gpfs/u/home/LLLD/LLLDashr/log_cdfsl_fewshot/deit_miniIN/slurm_err_logs/%j.err
#################

# time needed for job
#SBATCH --time=06:00:00
#################
# gpus per node
#SBATCH --gres=gpu:8
#################

# number of requested nodes
#SBATCH --nodes=4
#################

# slurm will send a signal this far out before it kills the job
#SBATCH --signal=USR1@150
#################

# Have SLURM send you an email when the job ends or fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=asrafulashiq@gmail.com

# #task per node
#SBATCH --ntasks-per-node=8
#################

# #cpu per task/gpu
#SBATCH --cpus-per-task=4
#################

# memory per cpu
#SBATCH --mem-per-cpu=10000
#################

export PYTHONFAULTHANDLER=1
export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)

srun python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_t_miniIN \
    --data-set mini-IN --epochs 7200
