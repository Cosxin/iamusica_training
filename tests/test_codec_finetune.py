import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch

from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import OnsetVelocityFrameDecoder
from ov_piano.utils import IncrementalHDF5


def _load_merge_module():
    path = Path(__file__).parents[1] / "merge_codec_hdf5.py"
    spec = importlib.util.spec_from_file_location("merge_codec_hdf5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(folder, name, delta):
    mel_path = folder / f"{name}_mel.h5"
    roll_path = folder / f"{name}_roll.h5"
    for path, height in ((mel_path, 3), (roll_path, 5)):
        with IncrementalHDF5(path, height) as output:
            output.append(
                np.full((height, 4), delta, dtype=np.float32),
                repr(("piece", "train")))
    return str(mel_path), str(roll_path)


def test_frame_head_shapes():
    model = OnsetsAndVelocities(
        2, 16, 8, conv1x1head=(8, 8), enable_frame_head=True)
    onsets, velocities, frames = model(torch.rand(2, 16, 20))
    assert len(onsets) == 3
    assert velocities.shape == frames.shape == (2, 8, 19)


def test_merge_repeats_matching_rolls(tmp_path):
    merge_module = _load_merge_module()
    clean = _pair(tmp_path, "clean", 1)
    aptx = _pair(tmp_path, "aptx", 2)
    sbc = _pair(tmp_path, "sbc", 3)
    out_mel = tmp_path / "mixed_mel.h5"
    out_roll = tmp_path / "mixed_roll.h5"
    provenance = tmp_path / "provenance.json"
    result = merge_module.merge(
        [("clean", *clean), ("aptx", *aptx), ("sbc", *sbc)],
        out_mel, out_roll, provenance)
    assert result["total_entries"] == 3
    with h5py.File(out_mel) as mel, h5py.File(out_roll) as roll:
        assert mel["data_idxs"].shape == roll["data_idxs"].shape == (2, 3)
        assert mel["metadata"][:].tolist() == roll["metadata"][:].tolist()
        assert [mel["data"][0, i * 4] for i in range(3)] == [1, 2, 3]


def test_frame_decoder_debounces_release_and_handles_reattack():
    decoder = OnsetVelocityFrameDecoder(
        1, nms_pool_ksize=3, frame_off_threshold=0.5,
        release_debounce_frames=2, min_note_frames=2)
    onsets = torch.zeros(1, 1, 14)
    onsets[0, 0, 1] = 0.9
    onsets[0, 0, 7] = 0.8
    velocities = torch.full_like(onsets, 0.75)
    frames = torch.zeros_like(onsets)
    frames[0, 0, 1:5] = 0.9
    frames[0, 0, 3] = 0.1  # one-frame flicker must not release
    frames[0, 0, 7:11] = 0.9

    notes = decoder(onsets, velocities, frames, pthresh=0.5)

    assert notes[["onset_idx", "offset_idx"]].values.tolist() == [
        [1, 5], [7, 11]]
    assert notes["release_reason"].tolist() == ["frame", "frame"]


def test_frame_decoder_closes_active_note_at_reattack():
    decoder = OnsetVelocityFrameDecoder(
        1, nms_pool_ksize=3, frame_off_threshold=0.5,
        release_debounce_frames=2, min_note_frames=2)
    onsets = torch.zeros(1, 1, 12)
    onsets[0, 0, 1] = 0.9
    onsets[0, 0, 6] = 0.8
    velocities = torch.ones_like(onsets)
    frames = torch.ones_like(onsets)

    notes = decoder(onsets, velocities, frames, pthresh=0.5)

    assert notes.iloc[0]["offset_idx"] == 6
    assert notes.iloc[0]["release_reason"] == "reattack"
