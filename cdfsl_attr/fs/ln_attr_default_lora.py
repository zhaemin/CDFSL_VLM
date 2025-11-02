import torch
import torch.nn.functional as F
import numpy as np

from contextlib import nullcontext
from fs.utils.model_utils import trainable_norm_params, num_params
from fs.utils.eval_utils import tokenize_texts, cls_acc, evaluate_attr

from loralib.utils import (
    mark_only_lora_as_trainable, apply_lora, get_lora_parameters, 
    lora_state_dict, save_lora, load_lora
)

import core.vision_encoder.transforms as transforms

import os
import matplotlib.pyplot as plt

import torch

def sinkhorn_batch(out, epsilon=0.05, n_iters=3):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py
    Q = torch.exp(out / epsilon) # Q is K-by-B for consistency with notations from our paper  => bs queries classes
    #print('1: ', torch.isnan(Q).any())
    B = Q.shape[2] # number of samples to assign -> num class
    K = Q.shape[1] # how many queries -> 8

    # make the matrix sums to 1
    sum_Q = torch.sum(Q, dim=(1,2), keepdim=True)
    Q /= sum_Q
    #print('2: ', torch.isnan(Q).any())

    for it in range(n_iters):
        # normalize each row: total weight per classes must be 1/Q
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        #print('3: ', torch.isnan(Q).any())

        # normalize each column: total weight per queries must be 1/C
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        #print('4: ', torch.isnan(Q).any())

    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q

def train_epoch(clip_model, optimizer, scheduler, scaler, train_loader, test_loader, dataset, tokenized_texts, count_iters, total_iters, epoch, args):
    clip_model.train()
    acc_train = 0
    tot_samples = 0
    loss_epoch = 0.
    subloss_epoch = 0.
    reg_loss_epoch = 0.

    text_context_manager = torch.no_grad if args.ln_modality == 'vision' else nullcontext
    vision_context_manager = torch.no_grad if args.ln_modality == 'text' else nullcontext

    num_classes = len(dataset.classnames)
    num_queries = args.num_attr
    
    if not hasattr(train_epoch, "attr_score_heatmap"):
        train_epoch.attr_score_heatmap = torch.zeros(num_classes, num_queries, device='cuda')
    
    for i, (images, target) in enumerate(train_loader):
        
        # move data to GPU
        images, target = images.cuda(), target.cuda()

        # get both text and image features
        # wrapping the forward pass in autocast
        with torch.amp.autocast('cuda', dtype=torch.float16):
            with text_context_manager():
                text_features = clip_model.encode_text(tokenized_texts)
            with vision_context_manager():
                image_features, attr_attn = clip_model.encode_image(images, return_attn=True, attr=True)
        
        # well, you know that clip normalizes the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # compute loss and backward with scaling
        cosine_similarity = torch.einsum('bnd, cd -> bnc', image_features, text_features)
        logits = clip_model.logit_scale.exp() * cosine_similarity[:, 0]
        attr_logits = cosine_similarity[:, 1:].float()

        #attr_sim = F.cosine_similarity(
        #    attr_logits.unsqueeze(2),  # [B, num_attr, 1, num_classes]
        #    attr_logits.unsqueeze(1),  # [B, 1, num_attr, num_classes]
        #    dim=-1
        #)
        #print(attr_sim[0])
        
        with torch.no_grad():
            attr_scores = sinkhorn_batch(attr_logits, epsilon=0.05).detach()
            B, N, C = attr_scores.size()
            attr_scores = attr_scores.gather(-1, target.unsqueeze(1).unsqueeze(-1).expand(-1, attr_scores.size(1), -1)).squeeze(-1) # B N
            
            for c in range(len(dataset.classnames)):
                mask = (target == c)
                if mask.any():
                    train_epoch.attr_score_heatmap[c] += attr_scores[mask].sum(dim=0)

        attr_losses = []
        attr_logits = attr_logits.transpose(0, 1) / 0.1
        
        for l in attr_logits:
            a_loss = F.cross_entropy(l, target, reduction='none') 
            attr_losses.append(a_loss.unsqueeze(1))
        
        attr_loss = (torch.cat(attr_losses, dim=1) * attr_scores).sum(dim=1).mean()
        loss = F.cross_entropy(logits, target) + attr_loss
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        optimizer.zero_grad()
        scaler.update()
        
        # compute accuracy and loss
        acc_train += cls_acc(logits, target) * target.shape[0]
        loss_epoch += loss.item() * target.shape[0]
        subloss_epoch += attr_loss.item() * target.shape[0]
        #reg_loss_epoch += reg_loss.item() * target.shape[0]
        tot_samples += target.shape[0]
        
        # check if we reached the total number of iterations
        count_iters += 1
        if count_iters == total_iters:
            break
        

    # print after each epoch
    if count_iters <= total_iters:
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        subloss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, Sub Loss: {:.4f}'.format(
            count_iters, total_iters, current_lr, acc_train, loss_epoch, subloss_epoch
            )
        )
        if epoch % 10 == 0:
            acc_test, gloabl_acc_test, attr_acc_test = evaluate_attr(args, clip_model, test_loader, template=dataset.template[0], classnames=dataset.classnames)
            print("**** Test accuracy at iterations {} (all categories): {:.2f}. / global acc: {} / attr acc: {} ****".format(count_iters, acc_test, gloabl_acc_test, attr_acc_test))

    return clip_model, count_iters

def visualize_query_similarity(name, queries, save_dir='./vis'): 
    num_queries = queries.shape[0] # (num_queries, N)

    # 각 query를 1D 벡터로 flatten
    attn_flat = queries.detach().cpu()

    # --- 거리 행렬 계산 (L2 distance) ---
    dist_matrix = torch.cdist(attn_flat, attn_flat, p=2)  # (num_queries, num_queries)
    dist_matrix = dist_matrix.numpy()

    # --- 시각화 ---
    plt.figure(figsize=(6, 5))
    plt.imshow(dist_matrix, cmap='magma')
    plt.colorbar(label='Euclidean Distance')
    plt.title("Query Euclidean Distance")
    plt.xlabel("Query Index")
    plt.ylabel("Query Index")
    plt.xticks(range(num_queries))
    plt.yticks(range(num_queries))
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/query_sim_{name}.png")
    plt.close()
    
def get_trainable_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    import torch.nn as nn
    import numpy as np
    from loralib.layers import LoRALayer, PlainMultiheadAttentionLoRA, SelfAttentionLoRA, LinearLoRA, MultiheadAttentionLoRAWOQproj
    trainable_params = []

    model.visual.attn_pool.attn = MultiheadAttentionLoRAWOQproj(model.visual.attn_pool.attn, r=32, enable_lora=['k']).cuda()
    model.requires_grad_(False)

    for name, param in model.visual.attn_pool.attn.named_parameters():
        if 'lora' in name:
            param.requires_grad_(True)
    
    model.visual.attn_pool.attr_probe.requires_grad_(True)
    
    model.visual.attn_pool.probe = nn.Parameter(model.visual.attn_pool.attn.q_proj(model.visual.attn_pool.probe))
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

def run_ln_only_attr(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    total_iters = args.n_iters * args.shots
    clip_model.visual.attn_pool.init_attr_probe(args.num_attr)
    clip_model = clip_model.cuda().float()
    
    queries = torch.cat([clip_model.visual.attn_pool.probe, clip_model.visual.attn_pool.attr_probe], dim=1)
    print('pre-training: ', queries)
    
    # train only layer-norm instances
    trainable_params = get_trainable_params(
        clip_model, 
        modality=args.ln_modality, 
        vision_start=args.ln_vision_start,
        text_start=args.ln_text_start
    )
    
    queries = torch.cat([clip_model.visual.attn_pool.probe, clip_model.visual.attn_pool.attr_probe], dim=1)
    print('qproj to query: ', queries)
    
    print(f"Trainable parameters: {num_params(clip_model, trainable=True):,}")
    
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

    #if hasattr(train_epoch, "attr_score_heatmap"):
    #    heatmap = train_epoch.attr_score_heatmap
    #    heatmap = heatmap.transpose(0,1).cpu()
    #    np.save(f'./vis/query{args.num_attr}_selection_heatmap.npy', heatmap.numpy())
#
    #    plt.figure(figsize=(12, 5))
    #    plt.imshow(heatmap, cmap='viridis', aspect='auto')
    #    plt.colorbar(label='Selection Frequency')
    #    plt.xlabel('Class Index')
    #    plt.ylabel('Query Index')
    #    plt.title('Query Selection Weight per Class')
#
    #    os.makedirs('./vis', exist_ok=True)
    #    plt.savefig(f'./vis/query{args.num_attr}_selection_heatmap.png')
    #    plt.close()

    #if args.n_iters > 0:
    #    os.makedirs('./checkpoint', exist_ok=True)
    #    torch.save(clip_model.state_dict(), f"checkpoint/{args.exp_name}_{args.dataset}.pth")
    #    save_lora(args, [clip_model.visual.attn_pool.attn])
    #
    #if args.n_iters == 0 and args.checkpoint != None:
    #    print("loading checkpoint")
    #    state_dict = torch.load(f'checkpoint/{args.checkpoint}_{args.dataset}.pth', map_location='cuda')
    #    clip_model.load_state_dict(state_dict, strict=False)
    #    load_lora(args, [clip_model.visual.attn_pool.attn])

    queries = torch.cat([clip_model.visual.attn_pool.probe, clip_model.visual.attn_pool.attr_probe], dim=1)
    print('after_trained: ', queries)
    
    # evaluate on test sets after training
    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader
        
        # evaluation on base classes
        acc_test_base, attr_acc_test_base = evaluate_attr(
            args, clip_model, test_base_loader, template=dataset.template[0], classnames=dataset.test_classnames, visualize=True
        )
        print("**** Test-Base accuracy: {:.2f}. / attr_acc: {} ****\n".format(acc_test_base, attr_acc_test_base))

        # evaluation on novel classes
        acc_test_novel, attr_acc_test_novel  = evaluate_attr(
            args, clip_model, test_new_loader, template=dataset.template[0], classnames=dataset.test_new_classnames, visualize=True
        )
        print("**** Test-Novel accuracy: {:.2f}. / attr_acc: {} ****\n".format(acc_test_novel, attr_acc_test_novel))
        result = {"acc_test_base": acc_test_base, "acc_test_new": acc_test_novel}
    
    else:
        acc_test, gloabl_acc_test, attr_acc_test = evaluate_attr(
            args, clip_model, test_loader, template=dataset.template[0], classnames=dataset.test_classnames, visualize=True
        )
        print("\n**** Final test accuracy (all categories): {:.2f}. / global_acc: {} / attr_acc: {} ****\n".format(acc_test, gloabl_acc_test, attr_acc_test))
        result = {"acc_test": acc_test}

    return result   
