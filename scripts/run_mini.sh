PLATFORM=$(uname -p)

if [ $PLATFORM = "x86_64" ]; then
    gpus=8
    node=4
else
    gpus=6
    node=5
fi

SLURM="--nodes ${nodes} --ngpus ${gpus}"

cmd="python run_with_submitit.py ${SLURM} \
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_t_mini \
    --data-set mini-IN --epochs 7200 --eval-type few_shot"

echo "$cmd"
eval "$cmd"
