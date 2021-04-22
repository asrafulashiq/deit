PLATFORM=$(uname -p)

if [ $PLATFORM = "x86_64" ]; then
    gpus=8
    nodes=4
else
    gpus=6
    nodes=5
fi

SLURM="--nodes ${nodes} --ngpus ${gpus}"

cmd="python run_with_submitit.py ${SLURM} \
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_t_cifar10 \
    --data-set CIFAR10 --epochs 7200 --eval-type linear --job_name deit_t_cifar10"

echo "$cmd"
eval "$cmd"
