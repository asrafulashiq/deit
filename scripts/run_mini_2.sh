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
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_tiny_mini \
    --data-set mini-IN --epochs 7200 --eval-type few_shot --job_name deit_tiny_mini"

echo "$cmd"
eval "$cmd"
