import torch
import torch.nn as nn
import clip


def named_modules_with_index(clip_model: nn.Module):
    assert hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer") and hasattr(clip_model, "transformer"), \
        "The model should have both vision and text transformer modules! RN not supported, implement it yourself :)"
    total_vision_blocks = len(clip_model.visual.transformer.resblocks)
    total_text_blocks = len(clip_model.transformer.resblocks)
    for name, module in clip_model.named_modules():
        if "ln_post" in name:
            yield name, module, total_vision_blocks
        if "ln_final" in name:
            yield name, module, total_text_blocks 
        if "ln_pre" in name:
            yield name, module, 0
        splitname = name.split('resblocks.')
        if len(splitname) == 1: # not a resblock
            yield name, module, -1
        else:
            block_idx = int(splitname[-1].split('.')[0])
            yield name, module, block_idx


def trainable_norm_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    trainable_params = []
    vision_trainable_params = []
    for name, module, block_idx in named_modules_with_index(model):
        curr_modality = 'vision' if 'visual' in name else 'text'
        curr_index = vision_start if curr_modality == 'vision' else text_start
        if isinstance(module, torch.nn.LayerNorm) and block_idx >= curr_index and (modality == 'both' or modality == curr_modality):
            trainable_params.extend(list(module.parameters()))
            if curr_modality == 'vision':
                vision_trainable_params.extend(list(module.parameters()))
            module.requires_grad_(True)
            print(f"Modality = {modality}, vision_start={vision_start}, text_start={text_start} ==> LayerNorm at {name} is trainable.")
        else:
            module.requires_grad_(False)
    return trainable_params, vision_trainable_params



def trainable_bias_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    trainable_params = []

    for param in model.parameters():
        param.requires_grad_(False)

    for name, module, block_idx in named_modules_with_index(model):
        curr_modality = 'vision' if 'visual' in name else 'text'
        curr_index = vision_start if curr_modality == 'vision' else text_start
        if hasattr(module, "bias") and block_idx >= curr_index and (modality == 'both' or modality == curr_modality):
            module.bias.requires_grad_(True)
            trainable_params.append(module.bias)
            print(f"Modality = {modality}, vision_start={vision_start}, text_start={text_start} ==> Bias at {name}.bias is trainable.")
    
    return trainable_params


def num_params(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())