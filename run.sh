mode="test"
while getopts "m:" opt; do
    case ${opt} in
    m)
        mode="$OPTARG"
        ;;
    esac
done

if [ $mode = "test" ]; then
    cmd="python  main.py --eval \
    --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224"
    echo "$cmd"
    eval "$cmd"
else
    cmd="python -m torch.distributed.launch \
    --nproc_per_node=2 --use_env main.py \
    --model deit_tiny_patch16_224 --batch-size 256 "
    echo "$cmd"
    eval "$cmd"
fi
