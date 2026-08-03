#!/usr/bin/env python
"""Train only the dedicated CC64 pedal module on frozen O&V features."""

import hashlib
import json
import os
import random
from ast import literal_eval
from dataclasses import asdict, dataclass
from typing import Optional

import h5py
from omegaconf import OmegaConf
import torch

from ov_piano import HDF5PathManager
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestroChunks
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.models.pedal import FrozenPedalModel
from ov_piano.pedal import pedal_targets_from_roll
from ov_piano.utils import IncrementalHDF5, load_model, set_seed


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED: int = 20260731
    MAESTRO_PATH: str = "datasets/maestro/maestro-v3.0.0"
    HDF5_MEL_PATH: str = "datasets/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5"
    HDF5_ROLL_PATH: str = "datasets/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"
    BASE_SNAPSHOT: str = "models/keyup_framehead_step12000.torch"
    OUTPUT_DIR: str = "runs/pedal_frozen"
    ALLOW_PARTIAL_HDF5: bool = False
    TRAIN_BS: int = 8
    TRAIN_BATCH_SECS: float = 2.5
    DATALOADER_WORKERS: int = 4
    ADAPTER_CHANNELS: int = 128
    HIDDEN_SIZE: int = 192
    GRU_LAYERS: int = 2
    DROPOUT: float = 0.15
    EVENT_RADIUS: int = 3
    STATE_POS_WEIGHT: float = 1.0
    EVENT_POS_WEIGHT: float = 8.0
    EVENT_LOSS_WEIGHT: float = 2.0
    LR: float = 0.001
    WEIGHT_DECAY: float = 0.0003
    MAX_STEPS: Optional[int] = None
    LOG_EVERY: int = 10
    SAVE_EVERY: int = 1000
    AMP: bool = True


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_basenames(path):
    with h5py.File(path, "r") as handle:
        return {
            literal_eval(item.decode("utf-8"))[0]
            for item in handle[IncrementalHDF5.METADATA_NAME]
        }


def save_checkpoint(path, model, optimizer, step, config, base_hash):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "base_sha256": base_hash,
    }, path)


if __name__ == "__main__":
    conf = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    set_seed(conf.RANDOM_SEED)
    os.makedirs(conf.OUTPUT_DIR, exist_ok=True)

    (_, sample_rate, _, hop_size, mel_bins, _, _) = \
        HDF5PathManager.parse_mel_hdf5_basename(os.path.basename(conf.HDF5_MEL_PATH))
    frames = round(conf.TRAIN_BATCH_SECS * sample_rate / hop_size)
    metadata = MetaMAESTROv3(conf.MAESTRO_PATH, splits=["train"],
                             years=MetaMAESTROv3.ALL_YEARS)
    if conf.ALLOW_PARTIAL_HDF5:
        available = available_basenames(conf.HDF5_MEL_PATH)
        metadata.data = [item for item in metadata.data
                         if os.path.basename(item[0]) in available]
    dataset = MelMaestroChunks(
        conf.HDF5_MEL_PATH, conf.HDF5_ROLL_PATH, frames, frames,
        *(item[0] for item in metadata.data), with_oob=False,
        as_torch_tensors=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.TRAIN_BS, shuffle=True,
        num_workers=conf.DATALOADER_WORKERS, pin_memory=True)

    base = OnsetsAndVelocities(2, mel_bins, 88, enable_frame_head=True)
    load_model(base, conf.BASE_SNAPSHOT, eval_phase=True, strict=True,
               to_cpu=conf.DEVICE == "cpu")
    model = FrozenPedalModel(
        base, conf.ADAPTER_CHANNELS, conf.HIDDEN_SIZE,
        conf.GRU_LAYERS, conf.DROPOUT).to(conf.DEVICE)
    model.train()
    frozen_before = {name: tensor.detach().cpu().clone()
                     for name, tensor in model.base.state_dict().items()}

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=conf.LR,
        weight_decay=conf.WEIGHT_DECAY)
    state_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        conf.STATE_POS_WEIGHT, device=conf.DEVICE))
    event_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        conf.EVENT_POS_WEIGHT, device=conf.DEVICE))
    scaler = torch.amp.GradScaler("cuda", enabled=conf.AMP and conf.DEVICE == "cuda")
    config = OmegaConf.to_container(conf)
    base_hash = sha256(conf.BASE_SNAPSHOT)
    with open(os.path.join(conf.OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as stream:
        json.dump({**config, "base_sha256": base_hash}, stream, indent=2)

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
                logits = model(logmels)
                loss_state = state_loss(logits[:, 0], targets[:, 0])
                loss_down = event_loss(logits[:, 1], targets[:, 1])
                loss_up = event_loss(logits[:, 2], targets[:, 2])
                loss = loss_state + conf.EVENT_LOSS_WEIGHT * (loss_down + loss_up)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step == 1 or step % conf.LOG_EVERY == 0:
                print(json.dumps({
                    "step": step, "loss": float(loss.detach()),
                    "state": float(loss_state.detach()),
                    "down": float(loss_down.detach()),
                    "up": float(loss_up.detach()),
                }), flush=True)
            if step % conf.SAVE_EVERY == 0:
                save_checkpoint(os.path.join(conf.OUTPUT_DIR, f"step-{step}.torch"),
                                model, optimizer, step, config, base_hash)

    for name, tensor in model.base.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), frozen_before[name]):
            raise RuntimeError(f"frozen base changed during training: {name}")
    save_checkpoint(os.path.join(conf.OUTPUT_DIR, f"smoke-step-{step}.torch"),
                    model, optimizer, step, config, base_hash)
    print(f"PEDAL_TRAINING_READY step={step} frozen_base_verified=true", flush=True)
