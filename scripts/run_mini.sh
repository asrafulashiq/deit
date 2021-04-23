PLATFORM=$(uname -p)

if [ $PLATFORM = "x86_64" ]; then
    gpus=8
    nodes=4
else
    gpus=4
    nodes=5
fi

SLURM="--nodes ${nodes} --ngpus ${gpus}"

cmd="python run_with_submitit.py ${SLURM} \
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_t_mini \
    --data-set mini-IN --epochs 7200 --eval-type few_shot --job_name deit_t_mini"

echo "$cmd"
eval "$cmd"

# python -m torch.distributed.launch --nproc_per_node=4 --use_en main.py \
#     --model deit_tiny_patch16_224 --batch-size 384 --output_dir outputs/deit_miniIN \
#     --warmup-epochs 40 --cooldown-epochs 80 --patience-epochs 80 --check_val_every_n_epoch 5 \
#     --data-set mini-IN --epochs 3000 --eval-type few_shot --num_episodes 200
