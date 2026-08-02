#!/usr/bin/env python
"""Fine-tune exact-timestamp CC64 confidence and sub-frame regression heads."""

import json
import math
import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf
import torch

from ov_piano import HDF5PathManager
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestroChunks
from ov_piano.eval import GtLoaderMaestro
from ov_piano.models.mobile_pedal import (
    EdgePedalAMTRegression,
    MultirateEdgePedalAMTRegression,
    OfflinePedalAMTRegression, OfflinePedalAMTRegressionFlatten,
    StreamingPedalAMTRegression, StreamingPedalAMTDualTimescale,
    StreamingPedalAMTDynamicCRF, StreamingPedalAMTDynamicCRFFlatten,
    StreamingPedalAMTDynamicCRFTwoStream)
from ov_piano.pedal import exact_event_targets
from ov_piano.crf import (
    crf_nll, state_unary_from_logit, transition_event_logits)
from ov_piano.utils import set_seed


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED: int = 20260801
    MAESTRO_PATH: str = "datasets/maestro/maestro-v3.0.0"
    HDF5_MEL_PATH: str = "datasets/maestro-mel.h5"
    HDF5_ROLL_PATH: str = "datasets/maestro-roll.h5"
    EVENT_CACHE_PATH: str = ""
    V1_CHECKPOINT: str = "runs/pedal-mobile-v1/final-step-12000.torch"
    FROM_SCRATCH: bool = False
    INIT_CHECKPOINT: str = ""
    OUTPUT_DIR: str = "runs/pedal-mobile-regression-v2"
    TRAIN_BS: int = 6
    TRAIN_BATCH_SECS: float = 5.0
    # Fixed event-output delay. An event output at t + D is supervised with
    # the target at t, giving the unidirectional recurrent trunk D seconds of
    # future audio without changing its parameter count. The CNN contributes
    # its existing ~96 ms local future receptive field in addition to this.
    FUTURE_CONTEXT_SECS: float = 0.0
    # Historical delayed-head runs shifted state supervision together with
    # events. New streaming models keep state at the current frame and delay
    # only the event heads, matching 4_eval_pedal.py and the live harness.
    DELAY_STATE_WITH_EVENTS: bool = True
    DATALOADER_WORKERS: int = 4
    MAX_STEPS: int = 6000
    SAVE_EVERY: int = 1000
    LOG_EVERY: int = 10
    EVENT_RADIUS: int = 3
    EVENT_POS_WEIGHT: float = 5.0
    CONFIDENCE_WEIGHT: float = 3.0
    OFFSET_WEIGHT: float = 2.0
    LR: float = 0.0005
    WARMUP_STEPS: int = 0
    MIN_LR_RATIO: float = 1.0
    WEIGHT_DECAY: float = 0.0003
    FREEZE_SHARED: bool = True
    FREEZE_STATE: bool = True
    MODEL_VARIANT: str = "baseline"
    SHARED_HIDDEN: int = 384
    TOWER_HIDDEN: int = 256
    SHARED_LAYERS: int = 2
    DROPOUT: float = 0.15
    BIDIRECTIONAL_HIDDEN: int = 224
    ORACLE_PROJECTION_SIZE: int = 160
    FRONTEND_SIZE: int = 512
    FAST_SIZE: int = 64
    CONTEXT_HIDDEN: int = 320
    CONTEXT_LAYERS: int = 2
    CONTEXT_POOL: int = 5
    EVENT_DELAY_FRAMES: int = 43
    HEAD_HIDDEN: int = 128
    STATE_WEIGHT: float = 0.0
    SLOW_DILATION: int = 2
    CRF_WEIGHT: float = 1.0
    TRANSITION_AUX_WEIGHT: float = 0.0
    TRANSIENT_AUX_WEIGHT: float = 0.0
    ALIGN_STATE_TO_EXACT_EVENTS: bool = False
    TRAIN_TRANSITION_HEAD_ONLY: bool = False
    AMP: bool = True
    # Keep the historical behavior by default.  The explicit switch and
    # differential LR make it possible to test whether the warm-started
    # acoustic representation is the limiting factor without destabilizing it.
    FREEZE_FRONTEND: bool = True
    FRONTEND_LR_SCALE: float = 0.1
    ALLOW_EXISTING_OUTPUT_DIR: bool = False


def transition_times(states, threshold=64):
    down, up, active = [], [], False
    for timestamp, value in states[["ts", "val"]].itertuples(index=False):
        next_active = value >= threshold
        if next_active != active:
            (down if next_active else up).append(float(timestamp))
        active = next_active
    return down, up


def load_transition_times(midi_path):
    _, sustain, _, _, _ = GtLoaderMaestro.get_midi_eventdata(midi_path)
    return transition_times(sustain)


class ExactPedalChunks(torch.utils.data.Dataset):
    def __init__(self, base, metadata, seconds_per_frame, radius,
                 event_cache_path="", align_state_to_events=False):
        self.base, self.seconds_per_frame, self.radius = base, seconds_per_frame, radius
        self.align_state_to_events = align_state_to_events
        self.file_starts = {}
        for index, (beg, _, _) in enumerate(base.data):
            name = base.metadata[base.metadata_chunks[index]][0]
            self.file_starts[name] = min(self.file_starts.get(name, beg), beg)
        names = list(self.file_starts)
        cache = None
        if event_cache_path and os.path.isfile(event_cache_path):
            cache = torch.load(event_cache_path, map_location="cpu")
            if cache.get("names") != names:
                raise ValueError("pedal event cache does not match training files")
            self.events = cache["events"]
        else:
            midi_paths = [metadata.get_file_abspath(name) + ".midi" for name in names]
            with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as executor:
                transitions = executor.map(load_transition_times, midi_paths, chunksize=4)
                self.events = dict(zip(names, transitions))
            if event_cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(event_cache_path)),
                            exist_ok=True)
                temporary = event_cache_path + f".tmp-{os.getpid()}"
                torch.save({"names": names, "events": self.events}, temporary)
                os.replace(temporary, event_cache_path)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        mel, roll, meta = self.base[index]
        name, beg = meta[0], meta[-3]
        chunk_start = (beg - self.file_starts[name]) * self.seconds_per_frame
        state = (roll[-3] >= 64).float()
        down, up = self.events[name]
        targets = exact_event_targets(
            state, down, up, chunk_start, self.seconds_per_frame,
            mel.shape[-1], self.radius, self.align_state_to_events)
        return mel, targets


def save(path, model, optimizer, step, config):
    model_type = ({
        "baseline": "mobile_regression",
        "dual_timescale": "mobile_dual_timescale",
        "dynamic_crf": "mobile_dynamic_crf",
        "dynamic_crf_flatten": "mobile_dynamic_crf_flatten",
        "dynamic_crf_two_stream": "mobile_dynamic_crf_two_stream",
        "offline_oracle": "mobile_offline_oracle",
        "offline_oracle_flatten": "mobile_offline_oracle_flatten",
        "edge_10ms": "mobile_edge_10ms",
        "edge_multirate_10ms": "mobile_edge_multirate_10ms",
    }[config.get("MODEL_VARIANT", "baseline")])
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": step, "config": config, "model_type": model_type}
    temporary = f"{path}.tmp-{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    conf = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    set_seed(conf.RANDOM_SEED)
    if os.path.exists(conf.OUTPUT_DIR) and not conf.ALLOW_EXISTING_OUTPUT_DIR:
        raise FileExistsError(f"refusing to overwrite {conf.OUTPUT_DIR}")
    os.makedirs(conf.OUTPUT_DIR, exist_ok=True)
    _, sample_rate, _, hop_size, mel_bins, _, _ = HDF5PathManager.parse_mel_hdf5_basename(
        os.path.basename(conf.HDF5_MEL_PATH))
    seconds_per_frame = hop_size / sample_rate
    frames = round(conf.TRAIN_BATCH_SECS / seconds_per_frame)
    future_frames = round(conf.FUTURE_CONTEXT_SECS / seconds_per_frame)
    if future_frames < 0 or future_frames >= frames:
        raise ValueError(
            "FUTURE_CONTEXT_SECS must be nonnegative and shorter than "
            "TRAIN_BATCH_SECS")
    metadata = MetaMAESTROv3(conf.MAESTRO_PATH, splits=["train"],
                             years=MetaMAESTROv3.ALL_YEARS)
    base = MelMaestroChunks(
        conf.HDF5_MEL_PATH, conf.HDF5_ROLL_PATH, frames, frames,
        *(item[0] for item in metadata.data), with_oob=False,
        as_torch_tensors=True)
    dataset = ExactPedalChunks(
        base, metadata, seconds_per_frame, conf.EVENT_RADIUS,
        conf.EVENT_CACHE_PATH, conf.ALIGN_STATE_TO_EXACT_EVENTS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.TRAIN_BS, shuffle=True,
        num_workers=conf.DATALOADER_WORKERS, pin_memory=True,
        persistent_workers=conf.DATALOADER_WORKERS > 0)
    if conf.FROM_SCRATCH:
        old = {key: conf[key] for key in
               ("SHARED_HIDDEN", "TOWER_HIDDEN", "SHARED_LAYERS", "DROPOUT")}
        v1 = None
    else:
        v1 = torch.load(conf.V1_CHECKPOINT, map_location="cpu")
        old = v1["config"]
    if conf.MODEL_VARIANT == "baseline":
        model = StreamingPedalAMTRegression(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"], shared_layers=old["SHARED_LAYERS"],
            dropout=old["DROPOUT"])
    elif conf.MODEL_VARIANT == "dual_timescale":
        model = StreamingPedalAMTDualTimescale(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"], shared_layers=old["SHARED_LAYERS"],
            dropout=old["DROPOUT"], slow_dilation=conf.SLOW_DILATION)
    elif conf.MODEL_VARIANT == "dynamic_crf":
        model = StreamingPedalAMTDynamicCRF(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"], shared_layers=old["SHARED_LAYERS"],
            dropout=old["DROPOUT"], slow_dilation=conf.SLOW_DILATION)
    elif conf.MODEL_VARIANT == "dynamic_crf_flatten":
        model = StreamingPedalAMTDynamicCRFFlatten(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"], shared_layers=old["SHARED_LAYERS"],
            dropout=old["DROPOUT"], slow_dilation=conf.SLOW_DILATION)
    elif conf.MODEL_VARIANT == "dynamic_crf_two_stream":
        model = StreamingPedalAMTDynamicCRFTwoStream(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"], shared_layers=old["SHARED_LAYERS"],
            dropout=old["DROPOUT"], slow_dilation=conf.SLOW_DILATION)
    elif conf.MODEL_VARIANT == "offline_oracle":
        model = OfflinePedalAMTRegression(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"], shared_layers=old["SHARED_LAYERS"],
            dropout=old["DROPOUT"],
            bidirectional_hidden=conf.BIDIRECTIONAL_HIDDEN)
    elif conf.MODEL_VARIANT == "offline_oracle_flatten":
        model = OfflinePedalAMTRegressionFlatten(
            mel_bins, shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"],
            shared_layers=old["SHARED_LAYERS"], dropout=old["DROPOUT"],
            bidirectional_hidden=conf.BIDIRECTIONAL_HIDDEN,
            projection_size=conf.ORACLE_PROJECTION_SIZE)
    elif conf.MODEL_VARIANT == "edge_10ms":
        model = EdgePedalAMTRegression(
            mel_bins, frontend_size=conf.FRONTEND_SIZE,
            shared_hidden=old["SHARED_HIDDEN"],
            tower_hidden=old["TOWER_HIDDEN"],
            shared_layers=old["SHARED_LAYERS"], dropout=old["DROPOUT"])
    elif conf.MODEL_VARIANT == "edge_multirate_10ms":
        model = MultirateEdgePedalAMTRegression(
            mel_bins, fast_size=conf.FAST_SIZE,
            context_hidden=conf.CONTEXT_HIDDEN,
            context_layers=conf.CONTEXT_LAYERS,
            context_pool=conf.CONTEXT_POOL,
            event_delay_frames=conf.EVENT_DELAY_FRAMES,
            head_hidden=conf.HEAD_HIDDEN, dropout=old["DROPOUT"])
    else:
        raise ValueError(f"unknown MODEL_VARIANT: {conf.MODEL_VARIANT}")
    copied = [] if v1 is None else model.load_v1(v1["model"])
    if conf.INIT_CHECKPOINT:
        initialized = torch.load(conf.INIT_CHECKPOINT, map_location="cpu")
        if conf.MODEL_VARIANT == "dual_timescale":
            model.load_regression(initialized["model"])
        elif conf.MODEL_VARIANT == "dynamic_crf":
            if initialized.get("model_type") == "mobile_dynamic_crf":
                model.load_state_dict(initialized["model"], strict=True)
            elif initialized.get("model_type") == "mobile_dual_timescale":
                model.load_dual_timescale(initialized["model"])
            else:
                model.load_regression(initialized["model"])
        elif conf.MODEL_VARIANT in ("dynamic_crf_flatten",
                                    "dynamic_crf_two_stream"):
            if initialized.get("model_type") != "mobile_dynamic_crf":
                raise ValueError(
                    "D7/D7b requires a mobile_dynamic_crf checkpoint")
            model.load_dynamic_crf(initialized["model"])
        else:
            model.load_state_dict(initialized["model"], strict=True)
    frozen_modules = []
    if conf.FREEZE_FRONTEND:
        frozen_modules.append(model.frontend)
    if conf.FREEZE_STATE:
        frozen_modules.append(model.towers[0])
    if conf.FREEZE_SHARED:
        frozen_modules.extend((model.shared, model.shared_norm))
    if conf.TRAIN_TRANSITION_HEAD_ONLY:
        if conf.MODEL_VARIANT != "dynamic_crf":
            raise ValueError(
                "TRAIN_TRANSITION_HEAD_ONLY requires MODEL_VARIANT=dynamic_crf")
        frozen_modules.extend((model.frontend, model.shared, model.shared_norm,
                               model.dual_timescale, model.towers))
    for module in frozen_modules:
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()
    model.to(conf.DEVICE)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not conf.FREEZE_FRONTEND:
        if conf.FRONTEND_LR_SCALE <= 0:
            raise ValueError("FRONTEND_LR_SCALE must be positive")
        frontend_ids = {id(parameter) for parameter in model.frontend.parameters()}
        frontend_parameters = [
            parameter for parameter in trainable if id(parameter) in frontend_ids]
        temporal_parameters = [
            parameter for parameter in trainable if id(parameter) not in frontend_ids]
        optimizer_parameters = [
            {"params": temporal_parameters, "lr": conf.LR},
            {"params": frontend_parameters,
             "lr": conf.LR * conf.FRONTEND_LR_SCALE},
        ]
    else:
        optimizer_parameters = trainable
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=conf.LR, weight_decay=conf.WEIGHT_DECAY)
    if conf.WARMUP_STEPS < 0 or conf.WARMUP_STEPS >= conf.MAX_STEPS:
        raise ValueError("WARMUP_STEPS must be in [0, MAX_STEPS)")
    if not 0 < conf.MIN_LR_RATIO <= 1:
        raise ValueError("MIN_LR_RATIO must be in (0, 1]")

    def lr_multiplier(completed_steps):
        if conf.WARMUP_STEPS and completed_steps < conf.WARMUP_STEPS:
            return (completed_steps + 1) / conf.WARMUP_STEPS
        decay_steps = conf.MAX_STEPS - conf.WARMUP_STEPS
        progress = ((completed_steps - conf.WARMUP_STEPS) / decay_steps
                    if decay_steps else 1.0)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return conf.MIN_LR_RATIO + (1.0 - conf.MIN_LR_RATIO) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    event_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        conf.EVENT_POS_WEIGHT, device=conf.DEVICE))
    state_loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=conf.AMP and conf.DEVICE == "cuda")
    config = OmegaConf.to_container(conf)
    config.update({key: old[key] for key in
                   ("SHARED_HIDDEN", "TOWER_HIDDEN", "SHARED_LAYERS", "DROPOUT")})
    with open(os.path.join(conf.OUTPUT_DIR, "config.json"), "w") as stream:
        json.dump({**config, "parameters": model.parameter_count,
                   "trainable_parameters": sum(p.numel() for p in trainable),
                   "initialized_tensors": len(copied)}, stream, indent=2)
    step = 0
    while step < conf.MAX_STEPS:
        for logmel, targets in loader:
            if step >= conf.MAX_STEPS:
                break
            logmel, targets = logmel.to(conf.DEVICE), targets.to(conf.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=conf.AMP and conf.DEVICE == "cuda"):
                transient_outputs = None
                if (conf.MODEL_VARIANT == "dynamic_crf_two_stream" and
                        conf.TRANSIENT_AUX_WEIGHT > 0):
                    outputs, transient_outputs = model.forward_with_transient(logmel)
                else:
                    outputs = model(logmel)
                if future_frames:
                    event_outputs = outputs[:, :, future_frames:]
                    event_targets = targets[:, :, :-future_frames]
                    if transient_outputs is not None:
                        transient_outputs = transient_outputs[:, :, future_frames:]
                else:
                    event_outputs, event_targets = outputs, targets
                down_conf = event_loss(event_outputs[:, 1], event_targets[:, 1])
                up_conf = event_loss(event_outputs[:, 3], event_targets[:, 3])
                if future_frames and not conf.DELAY_STATE_WITH_EVENTS:
                    state_loss = state_loss_fn(outputs[:, 0], targets[:, 0])
                else:
                    state_loss = state_loss_fn(
                        event_outputs[:, 0], event_targets[:, 0])
                offset_losses = []
                for conf_idx, offset_idx in ((1, 2), (3, 4)):
                    # Confidence is trained across the triangular window, but
                    # time displacement is a sub-frame target belonging only
                    # to the nearest frame (two frames only for an exact tie).
                    mask = ((event_targets[:, conf_idx] > 0) &
                            (event_targets[:, offset_idx].abs() <= 0.5))
                    pred = event_outputs[:, offset_idx].tanh()
                    offset_losses.append(torch.nn.functional.smooth_l1_loss(
                        pred[mask], event_targets[:, offset_idx][mask]) if mask.any()
                        else pred.sum() * 0)
                offset_loss = sum(offset_losses)
                loss = (conf.STATE_WEIGHT * state_loss +
                        conf.CONFIDENCE_WEIGHT * (down_conf + up_conf) +
                        conf.OFFSET_WEIGHT * offset_loss)
                transient_aux_loss = event_outputs.sum() * 0
                if transient_outputs is not None:
                    transient_down = event_loss(
                        transient_outputs[:, 0], event_targets[:, 1])
                    transient_up = event_loss(
                        transient_outputs[:, 2], event_targets[:, 3])
                    transient_offsets = []
                    for pred_idx, target_conf_idx, target_offset_idx in (
                            (1, 1, 2), (3, 3, 4)):
                        mask = ((event_targets[:, target_conf_idx] > 0) &
                                (event_targets[:, target_offset_idx].abs() <= 0.5))
                        pred = transient_outputs[:, pred_idx].tanh()
                        transient_offsets.append(
                            torch.nn.functional.smooth_l1_loss(
                                pred[mask], event_targets[:, target_offset_idx][mask])
                            if mask.any() else pred.sum() * 0)
                    transient_aux_loss = (
                        conf.CONFIDENCE_WEIGHT * (transient_down + transient_up) +
                        conf.OFFSET_WEIGHT * sum(transient_offsets))
                    loss = (loss + conf.TRANSIENT_AUX_WEIGHT *
                            transient_aux_loss)
                if conf.MODEL_VARIANT in (
                        "dynamic_crf", "dynamic_crf_flatten",
                        "dynamic_crf_two_stream"):
                    state_logits = event_outputs[:, 0]
                    transition_logits = event_outputs[:, 5:9].transpose(1, 2)
                    transition_logits = transition_logits.reshape(
                        *transition_logits.shape[:2], 2, 2)
                    state_targets = event_targets[:, 0].long()
                    # Keep the recurrent log-sum-exp in FP32 under AMP; a long
                    # sequence of half-precision normalizers is unnecessarily
                    # fragile and the two-state DP is computationally tiny.
                    structured_loss = crf_nll(
                        state_unary_from_logit(state_logits.float()),
                        transition_logits.float(), state_targets)
                    structured_loss = structured_loss / state_logits.shape[-1]
                    loss = loss + conf.CRF_WEIGHT * structured_loss
                    down_transition, up_transition = transition_event_logits(
                        transition_logits)
                    # transitions[:, 0] is unused by the linear-chain CRF, so
                    # do not supervise an output that can never affect decode.
                    transition_aux_loss = (
                        event_loss(down_transition[:, 1:],
                                   event_targets[:, 1, 1:]) +
                        event_loss(up_transition[:, 1:],
                                   event_targets[:, 3, 1:]))
                    loss = (loss + conf.TRANSITION_AUX_WEIGHT *
                            transition_aux_loss)
                else:
                    structured_loss = loss.detach() * 0
                    transition_aux_loss = loss.detach() * 0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step == 1 or step % conf.LOG_EVERY == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach()),
                    "lr": scheduler.get_last_lr()[0],
                    "state": float(state_loss.detach()),
                    "down_conf": float(down_conf.detach()),
                    "up_conf": float(up_conf.detach()),
                    "offset": float(offset_loss.detach()),
                    "crf": float(structured_loss.detach()),
                    "transition_aux": float(
                        transition_aux_loss.detach()),
                    "transient_aux": float(
                        transient_aux_loss.detach())}), flush=True)
            if step % conf.SAVE_EVERY == 0:
                save(os.path.join(conf.OUTPUT_DIR, f"step-{step}.torch"),
                     model, optimizer, step, config)
    save(os.path.join(conf.OUTPUT_DIR, f"final-step-{step}.torch"),
         model, optimizer, step, config)
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20
               if conf.DEVICE == "cuda" else 0.0)
    print(f"PEDAL_REGRESSION_TRAINING_COMPLETE step={step} "
          f"peak_cuda_mb={peak_mb:.1f}", flush=True)
