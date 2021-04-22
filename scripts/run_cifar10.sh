cmd="python run_with_submitit.py --nodes 1 --ngpus 2 \
    --model deit_tiny_patch16_224 --batch-size 256 --output_dir outputs/deit_t_cifar10 \
    --data-set CIFAR10 --epochs 7200 --eval-type linear"

echo "$cmd"
eval "$cmd"
