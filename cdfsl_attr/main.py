import clip
import time
import torch
import random
import argparse
import datetime
import numpy as np
import os

# custom imports
from datasets import build_dataset, build_dataloaders
from fs import run_cliplora, run_ln_only, run_twostage, run_twostage_attr, run_visonly, run_twostage_family, run_prompt_tuning, run_ln_only_attr, run_ln_only_attr_slot
from fs.utils import dump

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# helper message from the '--mode' argument to parse
MODE_HELPER = "Choose which experiment to run. Choices are:"
MODE_HELPER += "\t 1. 'cliplora': will run CLIP-LoRA as per https://arxiv.org/abs/2405.18541;"
MODE_HELPER += "\t 2. 'ln_only': will do FSA by only tuning layer-normalization instances according to --ln_modality and --norm_start;"
MODE_HELPER += "\t 3. (default) 'twostage': will run 2SFS, and you can customize it with the --peft, --n_iters and --n_iters_frac arguments;"


def reproducible_setup(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    
    # dataset arguments
    parser.add_argument('--root_path', type=str, default='../data/clip_fewshot', help="Root directory to where your datasets are stored.")
    parser.add_argument('--shots', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    
    # dataloading arguments
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--workers', type=int, default=8, help="num_workers for PyTorch Dataloaders")
    
    # model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    
    # training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)
    
    # experiment config
    parser.add_argument('--mode', type=str, default='twostage', choices=['cliplora', 'ln_only', 'twostage', 'ln_only_patchloss', 'lora_patchloss', 'ln_cls', 'onestage', 'twostage_attr', 'visonly', "twostage_family", "prompt_tuning", "ln_only_attr", "ln_only_attr_slot"],
                        help=MODE_HELPER)
    parser.add_argument('--setting', default='standard', type=str, choices=['standard', 'base2new'], 
                        help="Setting for the experiment. Set to 'standard' for all-to-all (train categories = eval categories) or 'base2new' otherwise.")
    parser.add_argument('--debug', default=False, type=int, 
                        help='Enable debugging mode (will run for a few iterations, then exit). Useful to check installation was successful.')
    parser.add_argument('--results_dir', type=str, default='results', help="Root folder to where your .csv results will be saved.")
    parser.add_argument('--exp_name', type=str, default='mycoolname', help="Experiment name (will be the basename of your .csv file).")

    # configs for 2sfs (default args will run main version)
    parser.add_argument('--n_iters', default=300, type=int, 
                        help='Shots Multiplier to get the total number of iterations. Denoted as M in the paper. Default=300.')
    parser.add_argument('--n_iters_frac', default=0.6, type=float,
                        help='Fraction of iterations to allocate to stage 1, denoted as alpha in the paper. Default=0.6')
    parser.add_argument('--peft', type=str, default='ln', choices=['ln', 'lora', 'bitfit', 'attnproj'], 
                        help='Parameter Efficient Fine-Tuning scheme to employ during the first stage when using 2SFS. Default: ln.')
    
    # additional configs you can explore for layer-normalization (leave default to reproduce results)
    parser.add_argument('--ln_modality', type=str, default='both', choices=['text', 'vision', 'both'],
                        help="Whether to tune LayerNorm instances in only the vision, text, or both encoders. Default: 'both'.")
    parser.add_argument('--ln_vision_start', default=0, type=int,
                        help="Whether to only start tuning LayerNorm instances after a certain block of the vision encoder;"
                         " Active if --ln_modality is 'both' or 'vision'. Default: 0 (tunes all instances).")
    parser.add_argument('--ln_text_start', default=0, type=int, 
                        help="Same as --ln_vision_start, but for the text encoder (therefore, active if --ln_modality is 'both' or 'text'). Default: 0.")
    
    # if you set --peft=lora or --mode=cliplora, then you may also wanna set these LoRA arguments 
    # NOTE: (all of the remaining args below are borrowed from CLIP-LoRA, no changes)
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3', 'custom1'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')

    # new arguments
    parser.add_argument('--dinoising', action='store_true')
    parser.add_argument('--teacher', type=str, default='dino')
    parser.add_argument('--target_layer', type=int, default=-1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--attr_num', type=int, default=8)

    
    args = parser.parse_args()
    return args


def main(args):
    time_in = time.time()
    reproducible_setup(args.seed)    
    
    # initialize clip model
    print("CLIP configs:", pe.CLIP.available_configs())
    # CLIP configs: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224', 'PE-Core-S16-384', 'PE-Core-T16-384']

    pe_model = pe.CLIP.from_config('PE-Core-B16-224', pretrained=True)  # Downloads from HF
    preprocess = transforms.get_image_transform(pe_model.image_size)
    pe_model.eval()

    #clip_model, preprocess = clip.load(args.backbone)
    #clip_model.eval()
    logit_scale = 100 # clip-lora uses this one
        
    # build the dataset (single obj with .train, .val and .test properties)
    dataset = build_dataset(
        dataset=args.dataset, 
        root_path=args.root_path, 
        shots=args.shots, 
        setting=args.setting,
        seed=args.seed
    )

    # create the dataloaders
    # NOTE: test_loader will be a tuple with both test_base and test_new loaders if we're evaluating the b2n setup
    train_loader, val_loader, test_loader = build_dataloaders(args, dataset, preprocess)

    # all functions in the fewshot module are exposing the same signature, so here we simplify fn calls with eval
    fn = eval(f"run_{args.mode}")
    fn_args = (args, pe_model, logit_scale, dataset, train_loader, val_loader, test_loader)
    res = fn(*fn_args)

    # print total time
    total_seconds = int(time.time() - time_in)
    print("Total time (hh:mm:ss) = ", str(datetime.timedelta(seconds=total_seconds)))
    
    # TODO: customize any logic to dump results on disk here =)
    res.update({"runtime": total_seconds})
    dump(res, vars(args), decimals=4)


if __name__ == '__main__':
    args = get_arguments()
    main(args)