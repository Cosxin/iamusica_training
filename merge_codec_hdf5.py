#!/usr/bin/env python
"""Merge aligned clean/codec IncrementalHDF5 pairs for Path-A training."""

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np

from ov_piano.utils import IncrementalHDF5


def _metadata(handle):
    return [ast.literal_eval(x.decode("utf-8")) for x in
            handle[IncrementalHDF5.METADATA_NAME]]


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path, include_hash):
    stat = os.stat(path)
    identity = {
        "path": os.path.abspath(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        identity["sha256"] = _sha256(path)
    return identity


def merge(variants, out_mel, out_roll, provenance_path, hash_inputs=False):
    """Merge ``[(name, mel_path, roll_path), ...]`` in deterministic order."""
    if len(variants) < 2:
        raise ValueError("At least two variants are required")
    for output in (out_mel, out_roll, provenance_path):
        if os.path.exists(output):
            raise FileExistsError(output)

    handles = []
    try:
        for name, mel_path, roll_path in variants:
            mel = h5py.File(mel_path, "r")
            roll = h5py.File(roll_path, "r")
            if _metadata(mel) != _metadata(roll):
                raise ValueError(f"{name}: mel/roll metadata mismatch")
            if not np.array_equal(mel[IncrementalHDF5.IDXS_NAME],
                                  roll[IncrementalHDF5.IDXS_NAME]):
                raise ValueError(f"{name}: mel/roll index mismatch")
            handles.append((name, mel_path, roll_path, mel, roll))

        reference_metadata = _metadata(handles[0][3])
        for name, _, _, mel, _ in handles[1:]:
            if _metadata(mel) != reference_metadata:
                raise ValueError(f"{name}: variant file order mismatch")

        mel_height = handles[0][3][IncrementalHDF5.DATA_NAME].shape[0]
        roll_height = handles[0][4][IncrementalHDF5.DATA_NAME].shape[0]
        with IncrementalHDF5(out_mel, mel_height) as mel_out, \
                IncrementalHDF5(out_roll, roll_height) as roll_out:
            for name, _, _, mel, roll in handles:
                for entry_idx, metadata in enumerate(reference_metadata):
                    mel_matrix, _ = IncrementalHDF5.get_element(mel, entry_idx)
                    roll_matrix, _ = IncrementalHDF5.get_element(
                        roll, entry_idx)
                    tagged = tuple(metadata) + (f"codec_variant={name}",)
                    tagged_str = repr(tagged)
                    mel_out.append(mel_matrix, tagged_str)
                    roll_out.append(roll_matrix, tagged_str)

        provenance = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ratio": {name: 1 for name, *_ in variants},
            "entries_per_variant": len(reference_metadata),
            "total_entries": len(reference_metadata) * len(variants),
            "inputs": [
                {"variant": name,
                 "mel": _file_identity(mel_path, hash_inputs),
                 "roll": _file_identity(roll_path, hash_inputs)}
                for name, mel_path, roll_path in variants
            ],
            "outputs": {"mel": out_mel, "roll": out_roll},
        }
        with open(provenance_path, "w", encoding="utf-8") as stream:
            json.dump(provenance, stream, indent=2)
            stream.write("\n")
        return provenance
    finally:
        for *_, mel, roll in handles:
            mel.close()
            roll.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", nargs=3, action="append", required=True,
        metavar=("NAME", "MEL_H5", "ROLL_H5"))
    parser.add_argument("--out-mel", required=True)
    parser.add_argument("--out-roll", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument(
        "--hash-inputs", action="store_true",
        help="SHA-256 every input (slow for full MAESTRO HDF5 files)")
    args = parser.parse_args()
    result = merge(args.variant, args.out_mel, args.out_roll, args.provenance,
                   hash_inputs=args.hash_inputs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
