ROOT_PATH=$1
NUM_ATTR=$2

for DATA in fgvc
do
    for SHOOT in 4
    do
        for SEED in 1
        do
        CUDA_VISIBLE_DEVICES=3 python main.py --shots ${SHOOT} --dataset ${DATA} --seed ${SEED} --mode 'ln_only_attr' --exp_name "ln_attr8" --root_path ${ROOT_PATH} --num_attr ${NUM_ATTR}
        done
    done
done