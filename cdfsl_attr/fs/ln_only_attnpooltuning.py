import torch
import torch.nn.functional as F

from contextlib import nullcontext
from fs.utils.model_utils import trainable_norm_params, num_params
from fs.utils.eval_utils import tokenize_texts, cls_acc, evaluate

import core.vision_encoder.transforms as transforms

import os
import matplotlib.pyplot as plt

def train_epoch(clip_model, optimizer, scheduler, scaler, train_loader, test_loader, dataset, tokenized_texts, count_iters, total_iters, epoch, args):
    clip_model.train()
    acc_train = 0
    tot_samples = 0
    loss_epoch = 0.

    text_context_manager = torch.no_grad if args.ln_modality == 'vision' else nullcontext
    vision_context_manager = torch.no_grad if args.ln_modality == 'text' else nullcontext

    for i, (images, target) in enumerate(train_loader):
        
        # move data to GPU
        images, target = images.cuda(), target.cuda()

        # get both text and image features
        # wrapping the forward pass in autocast
        with torch.amp.autocast('cuda', dtype=torch.float16):
            with text_context_manager():
                text_features = clip_model.encode_text(tokenized_texts)
            with vision_context_manager():
                image_features = clip_model.encode_image(images)
        
        # well, you know that clip normalizes the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # compute loss and backward with scaling
        cosine_similarity = clip_model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(cosine_similarity, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        optimizer.zero_grad()
        scaler.update()
        
        # compute accuracy and loss
        acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
        loss_epoch += loss.item() * target.shape[0]
        tot_samples += target.shape[0]
        
        # check if we reached the total number of iterations
        count_iters += 1
        if count_iters == total_iters:
            break
        
    # print after each epoch
    if count_iters <= total_iters:
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(
            count_iters, total_iters, current_lr, acc_train, loss_epoch
            )
        )
        if epoch % 10 == 0:
            acc_test = evaluate(args, clip_model, test_loader, template=dataset.template[0], classnames=dataset.classnames)
            print("**** Test accuracy at iterations {} (all categories): {:.2f}. ****".format(count_iters, acc_test))

    return clip_model, count_iters


def get_trainable_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    import torch.nn as nn
    import numpy as np
    from loralib.layers import LoRALayer, PlainMultiheadAttentionLoRA, SelfAttentionLoRA, LinearLoRA, MultiheadAttentionLoRAWOQproj
    trainable_params = []

    model.visual.attn_pool.attn = PlainMultiheadAttentionLoRA(model.visual.attn_pool.attn, r=32, enable_lora=['q', 'k']).cuda()
    model.requires_grad_(False)

    for name, param in model.visual.attn_pool.attn.named_parameters():
        if 'lora' in name:
            param.requires_grad_(True)

    model.visual.attn_pool.probe.requires_grad_(True)

    # layernorm param tuning
    model.visual.ln_pre.requires_grad_(True)
    model.visual.ln_post.requires_grad_(True)
    model.ln_final.requires_grad_(True)

    for name, module in model.named_modules():
        if ".resblocks." in name and ".ln_" in name:
            module.requires_grad_(True)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            trainable_params.append(param)

    return trainable_params


def run_ln_only(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    clip_model = clip_model.cuda().float()
    total_iters = args.n_iters * args.shots

    trainable_params = get_trainable_params(
        clip_model,
        modality=args.ln_modality, 
        vision_start=args.ln_vision_start,
        text_start=args.ln_text_start
    )

    print(f"Trainable parameters: {num_params(clip_model, trainable=True):,}")
    
    #optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    optimizer = torch.optim.AdamW([{'params' : trainable_params, 'lr' : args.lr }], lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))

    print(f"Using AdamW with lr={args.lr}, wd={args.wd}, betas=(0.9, 0.999).")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    # training 
    scaler = torch.amp.GradScaler('cuda')
    count_iters = 0

    # we only need to tokenize once
    tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
    tokenized_texts = tokenize_texts(template=dataset.template[0], classnames=dataset.classnames, tokenizer=tokenizer)

    # start training for a fixed number of gradient steps (total_iters)  
    epoch = 0
    while count_iters < total_iters:
        epoch += 1
        clip_model, count_iters = train_epoch(
            clip_model, 
            optimizer, 
            scheduler, 
            scaler, 
            train_loader,
            test_loader,
            dataset, 
            tokenized_texts, 
            count_iters, 
            total_iters,
            epoch,
            args
        )

        if args.debug: break

    if args.n_iters > 0:
        os.makedirs('./checkpoint', exist_ok=True)
        torch.save(clip_model.state_dict(), f"checkpoint/{args.exp_name}_{args.dataset}.pth")

    if args.n_iters == 0 and args.checkpoint != None:
        print("loading checkpoint")
        state_dict = torch.load(f'checkpoint/{args.checkpoint}_{args.dataset}.pth', map_location='cuda')
        clip_model.load_state_dict(state_dict, strict=False)

    # evaluate on test sets after training
    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader
        
        # evaluation on base classes
        acc_test_base = evaluate(
            args, clip_model, test_base_loader, template=dataset.template[0], classnames=dataset.test_classnames, visualize=True
        )
        print("**** Test-Base accuracy: {:.2f}. ****\n".format(acc_test_base))

        # evaluation on novel classes
        acc_test_novel = evaluate(
            args, clip_model, test_new_loader, template=dataset.template[0], classnames=dataset.test_new_classnames, visualize=True
        )
        print("**** Test-Novel accuracy: {:.2f}. ****\n".format(acc_test_novel))
        result = {"acc_test_base": acc_test_base, "acc_test_new": acc_test_novel}
    
    else:
        acc_test = evaluate(
            args, clip_model, test_loader, template=dataset.template[0], classnames=dataset.test_classnames, visualize=True
        )
        print("\n**** Final test accuracy (all categories): {:.2f}. ****\n".format(acc_test))
        result = {"acc_test": acc_test}

    return result
