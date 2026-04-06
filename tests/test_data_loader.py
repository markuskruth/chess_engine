# tests/test_data_loader.py — Phase 5 round-trip test
#
# Validates that data_loader.load_binary_game_data() correctly reads files
# written by the C++ SelfPlay::write_data() binary format.
#
# This test does NOT require a compiled C++ binary — it writes a synthetic
# .bin file using pure Python (matching the C++ format exactly) and checks
# that the loader recovers the original arrays.
#
# Run from the repo root:
#   python tests/test_data_loader.py

import struct
import tempfile
import os
import sys
import numpy as np

# Ensure repo root is on the path so data_loader is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import load_binary_game_data, load_into_buffer

STATE_CHANNELS = 20
BOARD_SZ       = 8
ACTION_DIM     = 4672

g_pass = 0
g_fail = 0


def check(condition: bool, msg: str) -> None:
    global g_pass, g_fail
    if condition:
        print(f"  [PASS] {msg}")
        g_pass += 1
    else:
        print(f"  [FAIL] {msg}", file=sys.stderr)
        g_fail += 1


def write_synthetic_bin(path: str, num_samples: int, rng: np.random.Generator):
    """Write a synthetic binary file in the C++ SelfPlay::write_data() format."""
    states   = rng.random((num_samples, STATE_CHANNELS, BOARD_SZ, BOARD_SZ), dtype=np.float64).astype(np.float32)
    policies = rng.random((num_samples, ACTION_DIM), dtype=np.float64).astype(np.float32)
    # Normalise policies so each row sums to 1 (as real game data would)
    policies = (policies / policies.sum(axis=1, keepdims=True)).astype(np.float32)
    values   = rng.uniform(-1.0, 1.0, size=(num_samples, 1)).astype(np.float32)

    with open(path, "wb") as f:
        # Header: uint32 num_samples, state_channels, board_size
        f.write(struct.pack("<III", num_samples, STATE_CHANNELS, BOARD_SZ))
        # Samples
        for i in range(num_samples):
            f.write(states[i].tobytes())    # 1280 floats
            f.write(policies[i].tobytes())  # 4672 floats
            f.write(values[i].tobytes())    # 1 float

    return states, policies, values


# ── Test 1: round-trip with typical sample count ──────────────────────────────

def test_round_trip():
    print("\n-- Round-trip: write synthetic data, read it back --")
    rng = np.random.default_rng(42)
    num_samples = 37  # odd size to catch alignment issues

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        orig_states, orig_policies, orig_values = write_synthetic_bin(path, num_samples, rng)
        states, policies, values = load_binary_game_data(path)

        check(states.shape   == (num_samples, STATE_CHANNELS, BOARD_SZ, BOARD_SZ),
              f"states shape == ({num_samples}, 20, 8, 8)")
        check(policies.shape == (num_samples, ACTION_DIM),
              f"policies shape == ({num_samples}, 4672)")
        check(values.shape   == (num_samples, 1),
              f"values shape == ({num_samples}, 1)")

        check(states.dtype   == np.float32, "states dtype == float32")
        check(policies.dtype == np.float32, "policies dtype == float32")
        check(values.dtype   == np.float32, "values dtype == float32")

        check(np.allclose(states,   orig_states,   atol=0.0, rtol=0.0),
              "states match bit-for-bit")
        check(np.allclose(policies, orig_policies, atol=0.0, rtol=0.0),
              "policies match bit-for-bit")
        check(np.allclose(values,   orig_values,   atol=0.0, rtol=0.0),
              "values match bit-for-bit")

        pi_sums = policies.sum(axis=1)
        check(np.allclose(pi_sums, 1.0, atol=1e-5),
              "all policy rows sum to ~1 after round-trip")

        check((values >= -1.0).all() and (values <= 1.0).all(),
              "all z values in [-1, 1]")
    finally:
        os.unlink(path)


# ── Test 2: empty file (0 samples) ────────────────────────────────────────────

def test_empty_file():
    print("\n-- Empty file (0 samples) --")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0, STATE_CHANNELS, BOARD_SZ))

        states, policies, values = load_binary_game_data(path)
        check(states.shape[0]   == 0, "states has 0 rows")
        check(policies.shape[0] == 0, "policies has 0 rows")
        check(values.shape[0]   == 0, "values has 0 rows")
    finally:
        os.unlink(path)


# ── Test 3: bad header (wrong channels) ───────────────────────────────────────

def test_bad_header():
    print("\n-- Bad header: wrong channel count --")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 1, 99, BOARD_SZ))  # channels=99

        raised = False
        try:
            load_binary_game_data(path)
        except ValueError:
            raised = True
        check(raised, "ValueError raised for wrong channel count")
    finally:
        os.unlink(path)


# ── Test 4: truncated file ────────────────────────────────────────────────────

def test_truncated_file():
    print("\n-- Truncated file --")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        # Write header claiming 10 samples but no data
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 10, STATE_CHANNELS, BOARD_SZ))

        raised = False
        try:
            load_binary_game_data(path)
        except ValueError:
            raised = True
        check(raised, "ValueError raised for truncated data")
    finally:
        os.unlink(path)


# ── Test 5: load_into_buffer integration ──────────────────────────────────────

def test_load_into_buffer():
    print("\n-- load_into_buffer integration with ReplayBuffer --")
    # Import here so test is skipped gracefully if utils not available
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils import ReplayBuffer
    except ImportError:
        print("  [SKIP] utils.ReplayBuffer not importable")
        return

    rng = np.random.default_rng(7)
    num_samples = 15

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        write_synthetic_bin(path, num_samples, rng)
        buf = ReplayBuffer(capacity=100)
        n = load_into_buffer(path, buf)

        check(n == num_samples,
              f"load_into_buffer returns {num_samples} (got {n})")
        check(len(buf) == num_samples,
              f"buffer size == {num_samples} after load")
        check(buf.states.shape[1:] == (STATE_CHANNELS, BOARD_SZ, BOARD_SZ),
              "buffer state shape correct")
    finally:
        os.unlink(path)


# ── Test 6: large sample count (stress) ───────────────────────────────────────

def test_large_file():
    print("\n-- Large file (1000 samples) --")
    rng = np.random.default_rng(99)
    num_samples = 1000

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        write_synthetic_bin(path, num_samples, rng)
        states, policies, values = load_binary_game_data(path)

        check(states.shape[0]   == num_samples, f"1000 states loaded")
        check(policies.shape[0] == num_samples, f"1000 policies loaded")
        check(values.shape[0]   == num_samples, f"1000 values loaded")
    finally:
        os.unlink(path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Phase 5: data_loader round-trip test ===")
    test_round_trip()
    test_empty_file()
    test_bad_header()
    test_truncated_file()
    test_load_into_buffer()
    test_large_file()
    print(f"\n=== Results: {g_pass} passed, {g_fail} failed ===")
    sys.exit(0 if g_fail == 0 else 1)
