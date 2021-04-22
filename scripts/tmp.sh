PLATFORM=$(uname -p)

gpus=1
nodes=1

SLURM="--nodes ${nodes} --ngpus ${gpus}"

cmd="python run_with_submitit.py ${SLURM} \
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/tmp --job_name tmp \
    --data-set CIFAR10 --epochs 7200 --eval-type linear --timeout 00:04:00 "

echo "$cmd"
eval "$cmd"
