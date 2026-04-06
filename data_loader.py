# data_loader.py — Phase 5: Python reader for C++ self-play binary data
#
# Reads files written by SelfPlay::write_data() / ParallelSelfPlay::write_data().
#
# File format (little-endian):
#   Header:  uint32 num_samples
#            uint32 state_channels   (always 20)
#            uint32 board_size       (always 8)
#   Per sample (repeated num_samples times):
#            float32[1280]  encoded state  (20 × 8 × 8)
#            float32[4672]  policy π
#            float32[1]     value target z
#
# Returns numpy arrays that can be fed directly into ReplayBuffer.add_batch()
# or converted to PyTorch tensors for training.

import numpy as np
from pathlib import Path

STATE_CHANNELS = 20
BOARD_SZ       = 8
ACTION_DIM     = 4672

_STATE_FLOATS  = STATE_CHANNELS * BOARD_SZ * BOARD_SZ  # 1280
_FLOATS_PER_SAMPLE = _STATE_FLOATS + ACTION_DIM + 1    # 5953


def load_binary_game_data(path: str):
    """
    Read a binary game-data file produced by C++ SelfPlay::write_data().

    Parameters
    ----------
    path : str or Path
        Path to the .bin file.

    Returns
    -------
    states   : np.ndarray, shape (N, 20, 8, 8), dtype float32
    policies : np.ndarray, shape (N, 4672),     dtype float32
    values   : np.ndarray, shape (N, 1),        dtype float32

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the header magic numbers are wrong or the file is truncated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Game data file not found: {path}")

    with open(path, "rb") as f:
        # ── Header (3 × uint32 = 12 bytes) ───────────────────────────────────
        header_bytes = f.read(12)
        if len(header_bytes) < 12:
            raise ValueError(f"{path}: file too small to contain a valid header")

        header = np.frombuffer(header_bytes, dtype="<u4")  # little-endian uint32
        num_samples, channels, board_size = int(header[0]), int(header[1]), int(header[2])

        if channels != STATE_CHANNELS:
            raise ValueError(
                f"{path}: expected {STATE_CHANNELS} state channels, got {channels}"
            )
        if board_size != BOARD_SZ:
            raise ValueError(
                f"{path}: expected board size {BOARD_SZ}, got {board_size}"
            )

        # Empty file is valid
        if num_samples == 0:
            return (
                np.zeros((0, STATE_CHANNELS, BOARD_SZ, BOARD_SZ), dtype=np.float32),
                np.zeros((0, ACTION_DIM),                          dtype=np.float32),
                np.zeros((0, 1),                                   dtype=np.float32),
            )

        # ── Sample data (read all at once for efficiency) ─────────────────────
        expected_floats = num_samples * _FLOATS_PER_SAMPLE
        raw_bytes = f.read(expected_floats * 4)
        if len(raw_bytes) != expected_floats * 4:
            actual = len(raw_bytes) // 4
            raise ValueError(
                f"{path}: truncated — expected {expected_floats} floats "
                f"({num_samples} samples), got {actual}"
            )

    raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(num_samples, _FLOATS_PER_SAMPLE)

    states   = raw[:, :_STATE_FLOATS].reshape(num_samples, STATE_CHANNELS, BOARD_SZ, BOARD_SZ).copy()
    policies = raw[:, _STATE_FLOATS : _STATE_FLOATS + ACTION_DIM].copy()
    values   = raw[:, _STATE_FLOATS + ACTION_DIM :].copy()  # shape (N, 1)

    return states, policies, values


def load_into_buffer(path: str, buffer):
    """
    Convenience wrapper: load a binary file and call buffer.add_batch().

    Parameters
    ----------
    path   : str or Path  — binary game data file
    buffer : ReplayBuffer — target buffer (must implement add_batch)

    Returns
    -------
    int  number of samples loaded
    """
    states, policies, values = load_binary_game_data(path)
    if len(states) > 0:
        buffer.add_batch(states, policies, values)
    return len(states)
