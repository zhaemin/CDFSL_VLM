import os
import torch
import os.path as osp
import json

import torch.nn.functional as F

import core.vision_encoder.transforms as transforms
import clip
from fs.utils.visualize import visualize_attentionmap, visualize_attentionbar, visualize_attention_combined
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sinkhorn_batch(out, epsilon=0.05, n_iters=3):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py
    Q = torch.exp(out / epsilon) # Q is K-by-B for consistency with notations from our paper  => bs queries classes
    B = Q.shape[2] # number of samples to assign -> num class
    K = Q.shape[1] # how many queries -> 8

    # make the matrix sums to 1
    sum_Q = torch.sum(Q, dim=(1,2), keepdim=True)
    Q /= sum_Q

    for it in range(n_iters):
        # normalize each row: total weight per queries must be 1/C
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per classes must be 1/Q
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
    
    with torch.no_grad():
        clip_weights = []
        texts = tokenize_texts(template, classnames, tokenizer).cuda()
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        clip_weights = class_embeddings.cuda()
    return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.to('cpu', non_blocking=True))
            labels.append(target.to('cpu', non_blocking=True))
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels


@torch.no_grad()
def zero_shot_eval(clip_model, dataset, loader, split="test"):
    assert split in ("train", "val", "test")
    # Textual features
    classnames = getattr(dataset, f"{split}_classnames") if split != "train" else dataset.classnames
    print("About to run clip_classifier", clip_model.visual.proj.device)
    textual_features = clip_classifier(classnames, dataset.template, clip_model)

    # Pre-load test features
    print("About to run pre_load_features", clip_model.visual.proj.device)
    test_features, test_labels = pre_load_features(clip_model, loader)
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = clip_model.logit_scale.exp() * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's {} accuracy: {:.2f}. ****\n".format(split, zs_acc))

    # free-up memory
    del test_features, test_labels, textual_features
    torch.cuda.empty_cache()
    return zs_acc



@torch.no_grad()
def evaluate(args, clip_model, loader, template, classnames, prompt=False, visualize=False):
    clip_model.eval()
    tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
    
    if prompt: # classname만 tokenize
        texts = [classname.replace('_', ' ') for classname in classnames]
        texts = tokenizer(classnames).cuda()
        #texts = clip.tokenize(classnames).cuda()
    else:
        texts = tokenize_texts(template=template, classnames=classnames, tokenizer=tokenizer)
        
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(texts, prompt=prompt)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features, attn_weights = clip_model.encode_image(images, return_attn=True)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = image_features @ text_features.t()
        acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
        tot_samples += len(cosine_similarity)

        # visualization
        if visualize:
            preds = cosine_similarity.argmax(dim=-1)
            if i <= 9:
                try:
                    #visualize_attentionbar(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False)
                    visualize_attention_combined(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False)
                except Exception as e:
                    print(e)
    
    acc /= tot_samples
    return acc

'''
@torch.no_grad()
def evaluate_attr(args, clip_model, loader, template, classnames, prompt=False, visualize=False):
    clip_model.eval()
    tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
    
    if prompt: # classname만 tokenize
        texts = [classname.replace('_', ' ') for classname in classnames]
        texts = tokenizer(classnames).cuda()
        #texts = clip.tokenize(classnames).cuda()
    else:
        texts = tokenize_texts(template=template, classnames=classnames, tokenizer=tokenizer)
        
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(texts, prompt=prompt)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    global_acc = 0.
    tot_samples = 0
    attr_mean_acc = 0.
    attr_acc = [0 for _ in range(args.num_attr)]

    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features, attn_weights = clip_model.encode_image(images, return_attn=True, attr=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        cosine_similarity = torch.einsum('bnd, cd -> bnc', image_features, text_features)

        #probs = F.softmax(cosine_similarity, dim=-1)
        #entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        global_logits = cosine_similarity[:, 0]
        logits = cosine_similarity.mean(dim=1)

        acc += cls_acc(logits, target) * len(logits)
        global_acc += cls_acc(global_logits, target) * len(logits)
        tot_samples += len(logits)

        # visualization
        if visualize:
            preds = logits.argmax(dim=-1)
            if i <= 9:
                visualize_attentionbar(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False, attr=True)
                visualize_attentionmap(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False, attr=True)
                #except Exception as e:
                #    print(e)
        # attr acc
        for k, l in enumerate(cosine_similarity[:, 1:].transpose(0, 1)):
            attr_acc[k] += cls_acc(l, target) * len(l)
    
    acc /= tot_samples
    global_acc /= tot_samples
    attr_acc = [(acc / tot_samples) for acc in attr_acc]
    return acc, global_acc, attr_acc
'''


@torch.no_grad()
def evaluate_attr(args, clip_model, loader, template, classnames, prompt=False, visualize=False):
    clip_model.eval()
    tokenizer = transforms.get_text_tokenizer(clip_model.context_length)

    if prompt:
        texts = [classname.replace('_', ' ') for classname in classnames]
        texts = tokenizer(classnames).cuda()
    else:
        texts = tokenize_texts(template=template, classnames=classnames, tokenizer=tokenizer)
        
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(texts, prompt=prompt)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    num_query = args.num_attr + 1
    num_class = len(classnames)
    query_class_correct = torch.zeros(num_query, num_class, device='cuda')
    query_class_total = torch.zeros(num_query, num_class, device='cuda')
    
    total_acc = 0.
    total_acc_attr = 0.
    tot_samples = 0

    all_logits = []
    all_preds = []
    all_targets = []

    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features, attn_weights = clip_model.encode_image(images, return_attn=True, attr=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        cosine_similarity = torch.einsum('bnd, cd -> bnc', image_features, text_features)
        pre_logits = cosine_similarity[:, 0].unsqueeze(dim=1)
        attr_logits = cosine_similarity[:, 1:]
        attr_scores = sinkhorn_batch(attr_logits, epsilon=0.05) # B N Cx

        attr_logits = attr_scores * attr_logits

        logits = torch.cat([pre_logits, attr_logits], dim=1)
        final_logits = logits.sum(dim=1)
        attr_logits = attr_logits.sum(dim=1)
        
        preds = logits.argmax(dim=-1)  # (B, N)

        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

        for q in range(num_query):
            for c in range(num_class):
                mask = (target == c)
                query_class_total[q, c] += mask.sum()
                query_class_correct[q, c] += (preds[mask, q] == c).sum()
        
        total_acc += cls_acc(final_logits, target) * len(target)
        total_acc_attr += cls_acc(attr_logits, target) * len(target)
        tot_samples += len(target)
        
        # visualization
        if visualize:
            if i == 0:
                plt.figure(figsize=(15, 6))
                sns.heatmap(attr_scores[0].cpu(), cmap="viridis", annot=False)
                plt.xlabel("Class index")
                plt.ylabel("Query index")
                plt.title("attribute scores per query-class")
                plt.tight_layout()
                plt.savefig(f"./vis/attr_heatmap_{args.exp_name}_target{classnames[target[0]]}.png", dpi=300)
                plt.close()
            if i <= 9:
                try:
                    preds = logits.sum(dim=1).argmax(dim=-1)
                    visualize_attentionbar(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False, attr=True)
                    visualize_attentionmap(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False, attr=True)
                except Exception as e:
                    print(e)

    all_logits = torch.cat(all_logits, dim=0)   
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)  

    acc_matrix = (query_class_correct / (query_class_total + 1e-8)).detach().cpu().numpy()
    overall_acc = (query_class_correct.sum(dim=1) / (query_class_total.sum(dim=1) + 1e-8)).detach().cpu().numpy()

    avg_acc = total_acc / tot_samples
    attr_acc = total_acc_attr / tot_samples

    #plt.figure(figsize=(1.2 * num_class, 0.6 * num_query))
    #sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="Blues",
    #            xticklabels=classnames, yticklabels=[f"Query {i}" for i in range(num_query)])
    #plt.title("Class-wise Query-wise Accuracy")
    #plt.xlabel("Class")
    #plt.ylabel("Query (Attribute)")
    #plt.tight_layout()
    #plt.savefig(f"./vis/{args.exp_name}_query_class_acc_heatmap.png")
    #plt.close()

    best_query_per_class = np.argmax(acc_matrix, axis=0)
    num_classes_per_query = np.bincount(best_query_per_class, minlength=num_query)
    best_preds = all_preds[torch.arange(len(all_targets)), torch.tensor(best_query_per_class)[all_targets]]
    oracle_acc = (best_preds == all_targets).float().mean().item()

    all_logits_tensor = all_logits.float()
    probs = torch.softmax(all_logits_tensor, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1) # B N
    low_entropy_idx = torch.argmin(entropy, dim=-1).cpu().numpy()  # 각 샘플별 low-entropy query

    # Oracle vs low-entropy query 선택 일치율
    match_ratio = (low_entropy_idx == torch.tensor(best_query_per_class)[all_targets].cpu().numpy()).mean()

    print("best query per class: ", best_query_per_class)
    print("num classes per queries: ", num_classes_per_query)
    print(f"Oracle-like accuracy (best query per class): {oracle_acc*100:.2f}%")
    print(f"Low-entropy query vs Oracle match ratio: {match_ratio*100:.2f}%")
    print(f"attr logit acc: {attr_acc}")

    return avg_acc, overall_acc[0], overall_acc[1:]

@torch.no_grad()
def evaluate_attr_multiattn(args, clip_model, loader, template, classnames, prompt=False, visualize=False):
    clip_model.eval()
    tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
    
    if prompt: # classname만 tokenize
        texts = [classname.replace('_', ' ') for classname in classnames]
        texts = tokenizer(classnames).cuda()
        #texts = clip.tokenize(classnames).cuda()
    else:
        texts = tokenize_texts(template=template, classnames=classnames, tokenizer=tokenizer)
        
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(texts, prompt=prompt)
    text_features = class_embeddings

    acc = 0.
    global_acc = 0.
    tot_samples = 0
    attr_mean_acc = 0.
    attr_acc = [0 for _ in range(args.num_attr)]

    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features, attn_weights = clip_model.encode_image(images, return_attn=True, attr=True)

        image_features_global = image_features[:, 0]
        image_features_attr = image_features[:, 1:]
        
        image_features_global = image_features_global / image_features_global.norm(dim=-1, keepdim=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            attr_prototype = F.scaled_dot_product_attention(text_features.unsqueeze(0).repeat(image_features.size(0), 1, 1), image_features_attr, image_features_attr)
        attr_prototype = attr_prototype / attr_prototype.norm(dim=-1, keepdim=True)
        
        logits = torch.einsum('bd, bcd -> bc', image_features_global, attr_prototype)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        global_cosine_similarity = image_features_global @ text_features.t()
        attr_cosine_similarity = torch.einsum('bnd, cd -> bnc', image_features_attr, text_features)
        
        acc += cls_acc(logits, target) * len(logits)
        global_acc += cls_acc(global_cosine_similarity, target) * len(logits)
        tot_samples += len(logits)

        # visualization
        if visualize:
            preds = logits.argmax(dim=-1)
            if i <= 9:
                try:
                    visualize_attentionbar(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False, attr=True)
                    visualize_attentionmap(images, attn_weights, preds, target, torch.arange(0, len(preds)), classnames, 1, args.dataset, args.exp_name, wrong=False, attr=True)
                except Exception as e:
                    print(e)
        # attr acc
        for k, l in enumerate(attr_cosine_similarity[:, 1:].transpose(0, 1)):
            attr_acc[k] += cls_acc(l, target) * len(l)
    
    acc /= tot_samples
    global_acc /= tot_samples
    attr_acc = [(acc / tot_samples) for acc in attr_acc]
    return acc, global_acc, attr_acc


def tokenize_texts(template, classnames, tokenizer, device='cuda'):
    texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
    tokenized_texts = tokenizer(texts).to(device)
    return tokenized_texts 

def tokenize_texts_with_CuPL(template, dataset, classnames, tokenizer, device='cuda'):
    prompts = []
    prompt_paths = [f'Prompt_CuPL/{dataset}.json']
    for prompt_path in prompt_paths:
        f = open(prompt_path)
        prompts.append(json.load(f))
    
    texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
    for prompt in prompts:
        for i, cls in enumerate(classnames):
            cls= cls.replace('_', ' ')
            for tp in prompt[cls]:
                texts[i] += tp
    tokenized_texts = tokenizer(texts).to(device)
    return tokenized_texts 

def dump(result: dict, args: dict, decimals: int = 4):
    import pandas as pd
    from typing import Iterable

    args["backbone"] = args["backbone"].replace("/", "-")
    
    outpath = osp.join(
        args["results_dir"], args["setting"], args["backbone"], args["dataset"], 
        f"shots_{args['shots']}", f"seed_{args['seed']}", args['mode'], 
        args["exp_name"]
    )
    if not outpath.endswith(".csv"): outpath += ".csv"
    os.makedirs(osp.dirname(outpath), exist_ok=True)

    result.update(args)
    result = {k: [v] for k, v in result.items()}
    df = pd.DataFrame.from_dict(result)

    for col in df.columns:
        if "acc" in str(col):
            df[col] = df[col].round(decimals)
    
    df.to_csv(outpath, index=False)
    print(f"Saved result at {outpath} =)")