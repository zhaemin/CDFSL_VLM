ROOT_PATH=$1

# imagenet sun397 fgvc eurosat stanford_cars food101 oxford_pets oxford_flowers caltech101 dtd ucf101
for DATA in fgvc
    do
    for SHOOT in 4
    do
        for SEED in 1
        do
            for ATTR in 2
            do
            CUDA_VISIBLE_DEVICES=3 python main.py --shots ${SHOOT} --dataset ${DATA} --seed ${SEED} --mode 'ln_only_attr' --exp_name "ln_attr${ATTR}_lora32_sinkhorn_rev3" --root_path "/home/haemin/study/data/clip_fewshot" --num_attr ${ATTR}
            done
        done
    done
done