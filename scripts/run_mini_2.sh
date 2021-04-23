PLATFORM=$(uname -p)

if [ $PLATFORM = "x86_64" ]; then
    gpus=4
    nodes=3
else
    gpus=4
    nodes=3
fi

SLURM="--nodes ${nodes} --ngpus ${gpus}"

cmd="python run_with_submitit.py ${SLURM} \
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_tiny_mini_less \
    --data-set mini-IN  --job_name deit_tiny_mini_less --no-model-ema \
    --warmup-epochs 30 --cooldown-epochs 40 --patience-epochs 60 --check_val_every_n_epoch 5 \
    --epochs 2000 --eval-type few_shot --num_episodes 200"

echo "$cmd"
eval "$cmd"
