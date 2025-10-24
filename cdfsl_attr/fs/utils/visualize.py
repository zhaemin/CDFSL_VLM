import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

import cv2

def denormalize(img_tensor):
    """
    img_tensor: (3, H, W), normalized image
    -> (H, W, 3) numpy, 0~1 범위
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-5), 0, 1)
    return img


def visualize_attentionbar(images, attn_weights, preds, target, total_indices, classnames: list[str],
                           max_samples: int, datasetname, exp_name, wrong, attr=False):
    shown = 0
    for idx in total_indices:
        if shown >= max_samples:
            return

        pred_class = classnames[preds[idx].item()]
        gt_class = classnames[target[idx].item()]

        # attn_weights shape: (B, reg+1, N)
        attns = attn_weights[idx]   # (reg+1, N)
        if not attr:
            attns = attns.unsqueeze(0)  # query 1개일 때

        num_queries = attns.shape[0]

        # --- 하나의 figure에 여러 bar chart (세로로 배치) ---
        fig, axes = plt.subplots(num_queries, 1, figsize=(10, 3 * num_queries), sharex=True)

        if num_queries == 1:
            axes = [axes]  # 단일 subplot일 때 리스트로 처리

        for i, attn in enumerate(attns):
            patch_attn = attn.detach().cpu().numpy().flatten()
            num_patches = len(patch_attn)
            patch_indices = np.arange(num_patches)

            ax = axes[i]
            ax.bar(patch_indices, patch_attn, color="skyblue", edgecolor="black")
            ax.axhline(y=patch_attn.mean(), color='red', linestyle='--', label=f"Mean={patch_attn.mean():.3f}")
            ax.legend()
            ax.set_ylabel(f"Q{i} Weight")
            ax.set_xlim(-1, num_patches)
            ax.set_ylim(0, max(0.01, patch_attn.max() * 1.1))

        axes[-1].set_xlabel("Patch Index")
        fig.suptitle(f"Patch Attention Weights\nGT: {gt_class}, Pred: {pred_class}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # --- 저장 ---
        save_dir = f'vis/attnmap2/{exp_name}/{datasetname}/{"wrong" if wrong else "right"}'
        os.makedirs(save_dir, exist_ok=True)
        safe_name = gt_class.replace("/", "_")
        plt.savefig(f'{save_dir}/attn_bar_{safe_name}_{idx}.png')
        plt.close()

        shown += 1


def visualize_attentionmap(images, attn_weights, preds, target, total_indices, classnames: list[str],
                           max_samples: int, datasetname, exp_name, wrong, attr=False):
    shown = 0
    for idx in total_indices:
        if shown >= max_samples:
            return

        img = denormalize(images[idx].cpu())  # (H, W, 3), 0~1
        pred_class = classnames[preds[idx].item()]
        gt_class = classnames[target[idx].item()]

        attns = attn_weights[idx]   # (reg+1, N)
        if not attr:
            attns = attns.unsqueeze(0)  # query 1개일 때

        num_queries = attns.shape[0]

        fig, axes = plt.subplots(1, num_queries, figsize=(4 * num_queries, 4))
        if num_queries == 1:
            axes = [axes]

        H, W, _ = img.shape

        for i, attn in enumerate(attns):
            num_patches = attn.shape[0]
            h = w = int(num_patches**0.5)

            # (1) attention map 계산
            patch_attn_map = attn[-(h*w):].reshape(h, w).detach().cpu().numpy()

            # (2) 정규화
            patch_attn_map = (patch_attn_map - patch_attn_map.min()) / (patch_attn_map.max() - patch_attn_map.min() + 1e-6)

            # (3) 업샘플 + 색상 적용
            heatmap = cv2.resize(patch_attn_map, (W, H))
            heatmap_color = plt.cm.jet(heatmap)[..., :3]  # RGBA 중 RGB만 사용

            # (4) 원본 이미지와 overlay
            overlay = 0.5 * img + 0.5 * heatmap_color
            overlay = np.clip(overlay, 0, 1)

            axes[i].imshow(overlay)
            axes[i].set_title(f"Query {i}")
            axes[i].axis("off")

        fig.suptitle(f"Attention Map Overlay\nGT: {gt_class}, Pred: {pred_class}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save_dir = f'vis/attnmap2/{exp_name}/{datasetname}/{"wrong" if wrong else "right"}'
        os.makedirs(save_dir, exist_ok=True)
        safe_name = gt_class.replace("/", "_")
        plt.savefig(f'{save_dir}/attentionmap_{safe_name}_{idx}.png')
        plt.close()

        shown += 1


def visualize_attention_combined(images, attn_weights, preds, target, total_indices, classnames: list[str],
                                 max_samples: int, datasetname, exp_name, wrong, attr=False):
    shown = 0
    for idx in total_indices:
        if shown >= max_samples:
            return

        img = denormalize(images[idx].cpu())  # (H, W, 3), 0~1
        pred_class = classnames[preds[idx].item()]
        gt_class = classnames[target[idx].item()]

        # attn_weights shape: (B, reg+1, N)
        attns = attn_weights[idx]  # (reg+1, N)
        if not attr:
            attns = attns.unsqueeze(0)  # query 1개일 때

        for i, attn in enumerate(attns):
            patch_attn = attn.detach().cpu().numpy().flatten()
            num_patches = len(patch_attn)
            h = w = int(num_patches ** 0.5)
            patch_indices = np.arange(num_patches)

            # --- Patch attention map ---
            patch_attn_map = patch_attn[-(h*w):].reshape(h, w)
            patch_attn_map = (patch_attn_map - patch_attn_map.min()) / (patch_attn_map.max() - patch_attn_map.min() + 1e-6)

            # 원본 이미지 크기로 업샘플
            H, W, _ = img.shape
            heatmap = cv2.resize(patch_attn_map, (W, H))
            heatmap_color = plt.cm.jet(heatmap)[..., :3]  # RGB
            overlay = 0.5 * img + 0.5 * heatmap_color

            # --- 시각화 ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # (1) Attention heatmap overlay
            axes[0].imshow(overlay)
            axes[0].set_title(f"Attention Map\nGT: {gt_class}, Pred: {pred_class}")
            axes[0].axis("off")

            # (2) Patch Attention Weight Bar Chart
            axes[1].bar(patch_indices, patch_attn, color="skyblue", edgecolor="black")
            axes[1].set_title("Patch-wise Attention Weights")
            axes[1].set_xlabel("Patch Index")
            axes[1].set_ylabel("Attention Weight")
            axes[1].set_xlim(-1, num_patches)
            axes[1].axhline(y=patch_attn.mean(), color='red', linestyle='--', label=f"Mean={patch_attn.mean():.3f}")
            axes[1].legend()

            plt.tight_layout()

            # --- 저장 ---
            save_dir = f'vis/attnmap2/{exp_name}/{datasetname}/{"wrong" if wrong else "right"}'
            os.makedirs(save_dir, exist_ok=True)
            safe_name = gt_class.replace("/", "_")
            plt.savefig(f'{save_dir}/attn_combined_{safe_name}_{idx}_q{i}.png')
            plt.close()

        shown += 1


def visualize_sim_with_text(backbone, images, classifier, preds, target, total_indices, classnames: list[str], max_samples: int, datasetname, exp_name, wrong):
    shown = 0
    for idx in total_indices:
        if shown >= max_samples:
            return
        
        #if classnames[target[idx].item()] not in ['Beagle', 'Border Collie', 'Alaskan Malamute']:
        #    return
        
        gt_idx = target[idx]
        gt_class = classnames[target[idx].item()]
        pred_class = classnames[preds[idx].item()]

        with torch.no_grad():
            img_features, _ = backbone(images[idx].unsqueeze(0), return_patch=True)
        
        img_features = img_features.squeeze() # n d
        text_features = classifier[gt_idx].unsqueeze(0) # 1 d 

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_patch = torch.einsum('nd, kd -> nk', img_features, text_features)
        h = w = 14

        # --- Patch attention (CLS 제외) ---
        patch_similarity_map = logits_patch[1:].reshape(h, w).detach().cpu().numpy()

        # --- 시각화 ---
        img = denormalize(images[idx].cpu())
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # (1) 원본 이미지
        axes[0].imshow(img)
        axes[0].set_title(f"Original\nGT: {gt_class}\n Pred: {pred_class}")
        axes[0].axis("off")

        # (2) Overlay Heatmap
        axes[1].imshow(img)
        im = axes[1].imshow(
            patch_similarity_map, 
            cmap="bwr", 
            alpha=0.5,   # 투명도
            extent=(0, img.shape[1], img.shape[0], 0)  # 이미지 크기에 맞게 맵핑
        )
        axes[1].set_title("Patch similarity with text")
        axes[1].axis("off")

        # Colorbar 추가
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # --- 저장 ---
        if wrong:
            save_dir = f'vis/patchsimmap2{exp_name}/{datasetname}/wrong'
        else:
            save_dir = f'vis/patchsimmap2/{exp_name}/{datasetname}/right'
        os.makedirs(save_dir, exist_ok=True)
        safe_name = gt_class.replace("/", "_")
        plt.savefig(f'{save_dir}/patchsimmap_{safe_name}_{idx}.png')
        plt.close()

        shown += 1



def plot_lowacc_topk_confusion(confusion_matrix, classnames, datasetname, save_dir="./vis"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"confusion_lowacc_top5_{datasetname}.png")

    cm_np = confusion_matrix.numpy()
    num_classes = len(classnames)

    # 클래스별 전체 샘플 수, 맞춘 개수
    total_per_class = cm_np.sum(axis=1)
    correct_per_class = np.diag(cm_np)

    # 정확도 계산 (0으로 나눔 방지)
    acc_per_class = np.divide(correct_per_class, total_per_class, out=np.zeros_like(correct_per_class, dtype=float), where=total_per_class!=0)

    # 정확도 낮은 top-5 클래스 인덱스
    top_idx = np.argsort(acc_per_class)[:10]

    # confusion matrix subset (행은 top-5만, 열은 전체)
    cm_top = cm_np[top_idx, :]
    classes_topk = [classnames[i] for i in top_idx]

    # row-wise 정규화 (비율)
    cm_topk_norm = cm_top.astype("float") / cm_top.sum(axis=1, keepdims=True)
    cm_topk_norm = np.nan_to_num(cm_topk_norm)

    # Heatmap 그리기
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        cm_topk_norm,
        annot=False, cmap="Blues",
        xticklabels=classnames,
        yticklabels=classes_topk,
        cbar=True
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True (Top-10 lowest accuracy)", fontsize=12)
    plt.title(f"Confusion Matrix (Top-10 lowest accuracy classes vs all preds) - {datasetname})", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] {save_path}")


def tsne_text(model, dataset, classnames):
    with torch.no_grad():
        # feats_2d: TSNE로 변환된 feature (N, 2)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        feats_2d = tsne.fit_transform(model.classifier.cpu().numpy())

        # --- variant -> family 매핑 적용 ---
        families = [dataset.variant_to_family[c] for c in classnames]  # variant 라벨을 family로 변환
        unique_families = sorted(list(set(families)))
        family_to_idx = {f: i for i, f in enumerate(unique_families)}
        numeric_labels = np.array([family_to_idx[f] for f in families])

        # --- 색상 팔레트 생성 ---
        num_families = len(unique_families)
        colors = plt.cm.get_cmap("tab20b", num_families)  # family 개수만큼 색상 뽑기

        # --- 시각화 ---
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(
            feats_2d[:, 0], feats_2d[:, 1],
            c=numeric_labels, cmap=colors, s=5
        )

        # legend 직접 생성 (family 기준)
        handles = [
            mpatches.Patch(color=colors(i), label=family)
            for i, family in enumerate(unique_families)
        ]

        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
        plt.savefig("vis/tsne_visualization_familywise_after2sfs.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_tsne(prototypes, num_classes, target_classes=None):
    """
    prototypes: [num_layers, num_classes, feat_dim]
    target_classes: 그릴 클래스 리스트 (예: [0, 3, 5])
                   None이면 모든 클래스 그림
    """
    num_layers, _, feat_dim = prototypes.shape

    # [num_layers * num_classes, feat_dim]
    X = prototypes.reshape(-1, feat_dim)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_2d = tsne.fit_transform(X)
    X_2d = X_2d.reshape(num_layers, num_classes, 2)

    plt.figure(figsize=(10, 8))
   

    # 그릴 class 선택
    if target_classes is None:
        target_classes = list(range(num_classes))
    
    colors = plt.cm.get_cmap("tab20", len(target_classes))

    for c in target_classes:
        xs = X_2d[:, c, 0]
        ys = X_2d[:, c, 1]

        # 레이어별 점
        for l in range(num_layers):
            plt.scatter(xs[l], ys[l],
                        color=colors(c),
                        alpha=(l+1)/num_layers,
                        s=60,
                        edgecolors='k')

        # trajectory 연결
        plt.plot(xs, ys, color=colors(c), alpha=0.8, linewidth=1.5)

    plt.title(f"t-SNE of Layer-wise Prototypes (Classes {target_classes})")
    plt.savefig(f'vis/Layer-wise_Class_Prototypes.png')
    plt.close()

