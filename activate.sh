export ARCH=$(uname -m)

if [ "$ARCH" = 'x86_64' ]; then
    conda activate fs_cdfsl
else
    conda activate fs_cdfsl
fi
