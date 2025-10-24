# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.optim.optimizer
from omegaconf import OmegaConf
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (get_model_state_dict,
                                                     get_state_dict,
                                                     set_state_dict)

from core.distributed import get_is_master

logger = logging.getLogger("CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")


@dataclass
class SaveEvery:
    every: int = 1000
    keep: int = 0


@dataclass
class CheckpointArgs:
    dump: SaveEvery = field(default_factory=SaveEvery)
    eval: SaveEvery = field(default_factory=SaveEvery)
    path: Optional[str] = None
    init_ckpt_path: Optional[str] = None
    vision_model_path: Optional[str] = None
    is_consolidated_model: bool = False
    continue_training_from_init: bool = False


def _get_key_step(name: str):
    return int(re.findall(RE_DIGITS, name)[-1])


def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path.mkdir(exist_ok=True)
        logger.info(f"Consolidating to: {str(consolidate_path)}")
        dcp_to_torch_save(ckpt_dir, str(consolidate_path / CONSOLIDATE_NAME))
        (consolidate_path / CONFIG_NAME).write_text(
            (Path(ckpt_dir) / CONFIG_NAME).read_text()
        )
        logger.info("Consolidated !")
    return consolidate_path


def load_from_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    if not (Path(ckpt_dir) / ".metadata").exists():
        raise ValueError(
            f"Please convert the checkpoint distcp format using `torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    state_dict = {}
    if optimizer is not None:
        state_dict[model_key], state_dict[optim_key] = get_state_dict(model, optimizer)
    else:
        state_dict[model_key] = get_model_state_dict(model)
        if model_key == "":  # If only loading a model directly, the key should be empty
            state_dict = state_dict.pop(model_key)

    dcp.load(state_dict, checkpoint_id=ckpt_dir)


class CheckpointManager:
    def __init__(self, args: CheckpointArgs):
        self.path = args.path
        self.dump_every = args.dump
        self.eval_every = args.eval
        self.init_ckpt_path = args.init_ckpt_path
        self.continue_training_from_init = args.continue_training_from_init

        assert os.path.exists(
            self.path
        ), f"Path {self.path} does not exist and needs to be created before using CheckpointManager (use instantiate_and_make_dir)"

        self.existing_saves = self.get_existing_saves()

    def get_existing_saves(self) -> List[Path]:
        folders = [
            p
            for p in Path(self.path).iterdir()
            if p.is_dir() and re.match(RE_FOLDER, p.name)
        ]
        folders.sort(key=lambda p: _get_key_step(p.name))
        return folders

    def clean_up(self):
        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_folders = []
        other_folders = []
        for p in self.existing_saves:
            is_dump = _get_key_step(p.name) % self.dump_every.every == 0
            is_eval = _get_key_step(p.name) % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval:
                eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval folders: {eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders = eval_folders[-self.eval_every.keep :]

        folder_to_keep = set(other_folders + dump_folders + eval_folders)
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if dist.get_rank() == 0:
            for folder in folder_to_remove:
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        assert file.name in [CONSOLIDATE_FOLDER]
                        for f in file.iterdir():
                            f.unlink()
                        file.rmdir()
                folder.rmdir()

        dist.barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(p.name))

    def get_last_step_path(self, dp_rank: int = 0) -> Optional[Path]:
        path = None
        for p in reversed(self.existing_saves):
            if (p / TRAIN_STATE_NAME.format(dp_rank)).is_file():
                path = p
                break
        return path

    def _create_folder(self, base_path: Path, folder_name: str) -> Path:
        folder = base_path / folder_name
        if get_is_master():
            folder.mkdir(parents=False, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return folder

    def _get_dp_tp_mesh(
        self, device_mesh: Optional[DeviceMesh] = None
    ) -> Tuple[int, int]:
        dp_rank = 0
        tp_rank = 0
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                dp_rank = device_mesh.get_local_rank("dp_replicate")
                if "dp_shard" in device_mesh.mesh_dim_names:
                    dp_rank = dp_rank * device_mesh[
                        "dp_replicate"
                    ].size() + device_mesh.get_local_rank("dp_shard")
            if "tp" in device_mesh.mesh_dim_names:
                tp_rank = device_mesh.get_local_rank("tp")
        return dp_rank, tp_rank

    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
    ):
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def save(
        self,
        model,
        optimizer,
        train_state,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> bool:

        # When creating directory check if only rank0 or is there other solution
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving to: {str(curr_save_dir)}")

        if dist.is_initialized():
            dist.barrier()

        logger.info("Saving...")
        state_dict = self.get_state_dict(model, optimizer)
        dcp.save(state_dict, checkpoint_id=curr_save_dir)
        logger.info("State dict saved!")

        if dist.is_initialized():
            dist.barrier()

        if get_is_master():
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                    indent=4,
                )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            logger.info(
                f"Saving train state to: {str(curr_save_dir / train_state_name)}"
            )
            # logger.info(f"train_state.state_dict()={train_state.state_dict()}")
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        self.existing_saves.append(curr_save_dir)

        self.clean_up()

        if dist.is_initialized():
            dist.barrier()
        return True

    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: Optional[Path] = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        logger.info("Reloading train state")
        with open(path / train_state_name, "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        dcp.load(state_dict, checkpoint_id=path)
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")

    @classmethod
    def instantiate_and_make_dir(cls, args: CheckpointArgs):
        if get_is_master():
            os.makedirs(args.path, exist_ok=True)
        dist.barrier()

        return cls(args)


def get_consolidated_ckpt_path(ckpt_dir: Path, mp_rank: int = 0, mp_size: int = 1):
    if mp_size == 1:
        assert mp_rank == 0
        no_rank_path = ckpt_dir / "consolidated.pth"
        if no_rank_path.exists():
            return no_rank_path
    return ckpt_dir / f"consolidated.{mp_rank:02d}.pth"


def load_consolidated_checkpoint(
    model: nn.Module,
    consolidated_path: str,
    vision_model_path: Optional[str] = None,
):
    """
    Loads a consolidated checkpoint into the model.
    This version supports both:
      - a single file named 'consolidated.pth'
      - multiple parts named like 'consolidated.00.pth', 'consolidated.01.pth', etc.
    """
    ckpt_path = Path(consolidated_path)
    cp_file = get_consolidated_ckpt_path(ckpt_path, mp_rank=0, mp_size=1)
    if cp_file.exists():
        # Use the single file
        st_dict = torch.load(cp_file, weights_only=True)
        if "model" in st_dict:
            st_dict = st_dict["model"]
    else:
        # Fall back to multi-part consolidated files (e.g. consolidated.00.pth, consolidated.01.pth, …)
        checkpoint_files = sorted(ckpt_path.glob("consolidated.*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No consolidated checkpoint file found in {ckpt_path}."
            )
        st_dict = {}
        for ckpt_file in checkpoint_files:
            part = torch.load(ckpt_file, weights_only=True)
            # If the checkpoint part is wrapped with "model", unwrap it
            if "model" in part:
                part = part["model"]
            # Merge the state dicts (assumes the keys are all unique or will correctly overwrite)
            st_dict.update(part)

    model.vision_projector.init_tensors()
    model.vision_model.init_tensors()
    model.rope_embeddings.reset_parameters()

    if vision_model_path is not None:
        model.vision_model.load_ckpt(vision_model_path)

    missing_keys, unexpected_keys = model.load_state_dict(st_dict, strict=False)
    missing_keys = [k for k in missing_keys if "tied_module.weight" not in k]
    if vision_model_path is not None:
        # vision_model is already loaded separately
        missing_keys = [k for k in missing_keys if "vision_model." not in k]
    if len(missing_keys) > 0:
        logger.warning(f"Missing keys when reloading: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected keys when reloading: {unexpected_keys}")
