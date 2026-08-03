#!/usr/bin/env python
"""Train the direct-logmel Mobile-AMT-inspired CC64 model."""

import hashlib
import json
import os
from ast import literal_eval
from dataclasses import asdict, dataclass
from typing import Optional

import h5py
from omegaconf import OmegaConf
import torch

from ov_piano import HDF5PathManager
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestroChunks
from ov_piano.models.mobile_pedal import StreamingPedalAMT
from ov_piano.pedal import pedal_targets_from_roll
from ov_piano.utils import IncrementalHDF5, set_seed


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED: int = 20260801
    MAESTRO_PATH: str = "datasets/maestro/maestro-v3.0.0"
    HDF5_MEL_PATH: str = "datasets/maestro-mel.h5"
    HDF5_ROLL_PATH: str = "datasets/maestro-roll.h5"
    OUTPUT_DIR: str = "runs/pedal-mobile-v1"
    TRAIN_BS: int = 6
    TRAIN_BATCH_SECS: float = 5.0
    DATALOADER_WORKERS: int = 4
    SHARED_HIDDEN: int = 384
    TOWER_HIDDEN: int = 256
    SHARED_LAYERS: int = 2
    DROPOUT: float = 0.15
    EVENT_RADIUS: int = 5
    EVENT_POS_WEIGHT: float = 5.0
    EVENT_LOSS_WEIGHT: float = 3.0
    CONSISTENCY_LOSS_WEIGHT: float = 0.5
    LR: float = 0.001
    WEIGHT_DECAY: float = 0.0003
    MAX_STEPS: Optional[int] = None
    STEPS_PER_EPOCH: Optional[int] = None
    LOG_EVERY: int = 10
    SAVE_EVERY: int = 1000
    AMP: bool = True
    ALLOW_EXISTING_OUTPUT_DIR: bool = False


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, step, config):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "config": config,
    }, path)


if __name__ == "__main__":
    conf = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    set_seed(conf.RANDOM_SEED)
    if os.path.exists(conf.OUTPUT_DIR) and not conf.ALLOW_EXISTING_OUTPUT_DIR:
        raise FileExistsError(f"refusing to overwrite {conf.OUTPUT_DIR}")
    os.makedirs(conf.OUTPUT_DIR, exist_ok=True)
    (_, sample_rate, _, hop_size, mel_bins, _, _) = \
        HDF5PathManager.parse_mel_hdf5_basename(os.path.basename(conf.HDF5_MEL_PATH))
    frames = round(conf.TRAIN_BATCH_SECS * sample_rate / hop_size)
    metadata = MetaMAESTROv3(
        conf.MAESTRO_PATH, splits=["train"], years=MetaMAESTROv3.ALL_YEARS)
    dataset = MelMaestroChunks(
        conf.HDF5_MEL_PATH, conf.HDF5_ROLL_PATH, frames, frames,
        *(item[0] for item in metadata.data), with_oob=False,
        as_torch_tensors=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.TRAIN_BS, shuffle=True,
        num_workers=conf.DATALOADER_WORKERS, pin_memory=True,
        persistent_workers=conf.DATALOADER_WORKERS > 0)
    model = StreamingPedalAMT(
        mel_bins, shared_hidden=conf.SHARED_HIDDEN,
        tower_hidden=conf.TOWER_HIDDEN, shared_layers=conf.SHARED_LAYERS,
        dropout=conf.DROPOUT).to(conf.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    state_loss_fn = torch.nn.BCEWithLogitsLoss()
    event_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        conf.EVENT_POS_WEIGHT, device=conf.DEVICE))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=conf.AMP and conf.DEVICE == "cuda")
    config = OmegaConf.to_container(conf)
    with open(os.path.join(conf.OUTPUT_DIR, "config.json"), "w") as stream:
        json.dump({**config, "parameters": model.parameter_count}, stream, indent=2)
    print(json.dumps({"parameters": model.parameter_count, "frames": frames}), flush=True)
    step = 0
    while conf.MAX_STEPS is None or step < conf.MAX_STEPS:
        for logmels, rolls, _ in loader:
            if conf.MAX_STEPS is not None and step >= conf.MAX_STEPS:
                break
            logmels = logmels.to(conf.DEVICE, non_blocking=True)
            targets = pedal_targets_from_roll(
                rolls.to(conf.DEVICE, non_blocking=True),
                event_radius=conf.EVENT_RADIUS)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=conf.AMP and conf.DEVICE == "cuda"):
                logits = model(logmels)[:, :, 1:]
                if logits.shape != targets.shape:
                    raise RuntimeError(f"shape mismatch {logits.shape} != {targets.shape}")
                state_loss = state_loss_fn(logits[:, 0], targets[:, 0])
                down_loss = event_loss_fn(logits[:, 1], targets[:, 1])
                up_loss = event_loss_fn(logits[:, 2], targets[:, 2])
                state_probability = logits[:, 0].sigmoid()
                delta = torch.nn.functional.pad(
                    state_probability[:, 1:] - state_probability[:, :-1], (1, 0))
                consistency = (
                    (torch.relu(delta) - logits[:, 1].sigmoid()).abs().mean() +
                    (torch.relu(-delta) - logits[:, 2].sigmoid()).abs().mean())
                loss = (state_loss + conf.EVENT_LOSS_WEIGHT *
                        (down_loss + up_loss) +
                        conf.CONSISTENCY_LOSS_WEIGHT * consistency)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step == 1 or step % conf.LOG_EVERY == 0:
                print(json.dumps({
                    "step": step, "loss": float(loss.detach()),
                    "state": float(state_loss.detach()),
                    "down": float(down_loss.detach()), "up": float(up_loss.detach()),
                    "consistency": float(consistency.detach()),
                }), flush=True)
            if step % conf.SAVE_EVERY == 0:
                save_checkpoint(os.path.join(
                    conf.OUTPUT_DIR, f"step-{step}.torch"),
                    model, optimizer, step, config)
    save_checkpoint(os.path.join(conf.OUTPUT_DIR, f"final-step-{step}.torch"),
                    model, optimizer, step, config)
    print(f"MOBILE_PEDAL_TRAINING_COMPLETE step={step}", flush=True)
