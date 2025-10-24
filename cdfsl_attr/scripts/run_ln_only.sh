for DATA in imagenet sun397 fgvc eurosat stanford_cars food101 oxford_pets oxford_flowers caltech101 dtd ucf101
do
    for SHOOT in 4
    do
        for SEED in 1
        do
        CUDA_VISIBLE_DEVICES=2 python main.py --shots ${SHOOT} --dataset ${DATA} --seed ${SEED} --mode 'ln_only' --exp_name "ln_only"
        done
    done
done
