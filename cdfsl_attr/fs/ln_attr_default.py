import torch
import torch.nn.functional as F

from contextlib import nullcontext
from fs.utils.model_utils import trainable_norm_params, num_params
from fs.utils.eval_utils import tokenize_texts, cls_acc, evaluate_attr

import core.vision_encoder.transforms as transforms

import os
import matplotlib.pyplot as plt


import torch

def kl_div(dists):
    B, M, N = dists.shape
    probs = F.softmax(dists, dim=-1)
    log_p = F.log_softmax(dists, dim=-1)

    entropy = torch.sum(probs * log_p, dim=2, keepdim=True)  # (B, M, 1)
    cross_entropy = probs @ log_p.transpose(1, 2)  # (B, M, M)
    
    kl_matrix = entropy - cross_entropy  # (B, M, M)

    mask = ~torch.eye(M, dtype=torch.bool, device=probs.device)
    mean_kl_per_batch = kl_matrix[:, mask].mean(dim=1)  # (B,)

    return mean_kl_per_batch.mean()


def train_epoch(clip_model, optimizer, scheduler, scaler, train_loader, test_loader, dataset, tokenized_texts, count_iters, total_iters, epoch, args):
    clip_model.train()
    acc_train = 0
    tot_samples = 0
    loss_epoch = 0.
    subloss_epoch = 0.
    reg_loss_epoch = 0.

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
                image_features, attr_attn = clip_model.encode_image(images, return_attn=True, attr=True)
        
        # well, you know that clip normalizes the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        #kl_div_loss = kl_div(attr_attn) * 1000

        # compute loss and backward with scaling
        cosine_similarity = clip_model.logit_scale.exp() * torch.einsum('bnd, cd -> bnc', image_features, text_features).mean(dim=1)
        
        loss = F.cross_entropy(cosine_similarity, target)# - kl_div_loss
        scaler.scale(loss).backward()

        clip_model.visual.attn_pool.attn.in_proj_weight.grad[768*2:] = 0
        clip_model.visual.attn_pool.attn.in_proj_bias.grad[768*2:] = 0

        scaler.step(optimizer)
        scheduler.step()
        optimizer.zero_grad()
        scaler.update()
        
        # compute accuracy and loss
        acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
        loss_epoch += loss.item() * target.shape[0]
        #subloss_epoch += kl_div_loss.item() * target.shape[0]
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
            count_iters, total_iters, current_lr, acc_train, loss_epoch, loss_epoch
            )
        )
        if epoch % 10 == 0:
            acc_test, gloabl_acc_test, attr_acc_test = evaluate_attr(args, clip_model, test_loader, template=dataset.template[0], classnames=dataset.classnames)
            print("**** Test accuracy at iterations {} (all categories): {:.2f}. / global acc: {} / attr acc: {} ****".format(count_iters, acc_test, gloabl_acc_test, attr_acc_test))

    return clip_model, count_iters



def run_ln_only_attr(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    total_iters = args.n_iters * args.shots
    
    clip_model = clip_model.cuda().float()
    print('pretrained: ', clip_model.visual.attn_pool.attr_probe)
    
    # train only layer-norm instances
    trainable_params, vision_trainable_params = trainable_norm_params(
        clip_model, 
        modality=args.ln_modality, 
        vision_start=args.ln_vision_start,
        text_start=args.ln_text_start
    )

    clip_model.visual.attn_pool.attn.in_proj_weight.requires_grad_(True)
    clip_model.visual.attn_pool.attn.in_proj_bias.requires_grad_(True)
    clip_model.visual.attn_pool.attr_probe.requires_grad_(True)

    new_trainable_params = []
    new_trainable_params.append(clip_model.visual.attn_pool.attn.in_proj_weight)
    new_trainable_params.append(clip_model.visual.attn_pool.attn.in_proj_bias)
    new_trainable_params.append(clip_model.visual.attn_pool.attr_probe)
    
    print(f"Trainable parameters: {num_params(clip_model, trainable=True):,}")
    
    optimizer = torch.optim.AdamW([{'params' : trainable_params, 'lr' : args.lr }, {'params' : new_trainable_params, 'lr' : 0.001}], lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
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


    print('after_trained: ', clip_model.visual.attn_pool.attr_probe)
    
    '''
    W = clip_model.visual.attn_pool.attn.in_proj_weight.detach().cpu()
    dim = clip_model.visual.width
    W_Q, W_K, W_V = W[:dim], W[dim:2*dim], W[2*dim:]

    b = clip_model.visual.attn_pool.attn.in_proj_bias.detach().cpu()
    b_Q, b_K, b_V = b[:dim], b[dim:2*dim], b[2*dim:]

    plt.figure(figsize=(10, 3))
    for i, (name, w) in enumerate(zip(["Q", "K", "V"], [W_Q, W_K, W_V])):
        plt.subplot(1, 3, i+1)
        plt.hist(w.flatten().numpy(), bins=100, color='gray')
        plt.title(f"{name} weight distribution")
        plt.xlabel("value")
        plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f'./vis/qkv_weight_attr_{args.exp_name}.png')
    plt.close()

    plt.figure(figsize=(10, 3))
    for i, (name, b_) in enumerate(zip(["Q", "K", "V"], [b_Q, b_K, b_V])):
        plt.subplot(1, 3, i+1)
        plt.hist(b_.flatten().numpy(), bins=100, color='orange')
        plt.title(f"{name} bias distribution")
        plt.xlabel("value")
        plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f'./vis/qkv_bias_attr_{args.exp_name}.png')
    plt.close()
    '''

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
