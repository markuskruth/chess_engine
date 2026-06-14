# CLAUDE.md

Guidance for working in this repository.

## What this project is

An **AlphaZero-style chess engine** trained by reinforcement learning through
self-play. It learns purely from self-play games (no human game database) using
a CNN policy/value network guided by Monte Carlo Tree Search (MCTS).

The system is a **hybrid C++/Python pipeline**:

- **C++** handles the compute-heavy *self-play data generation* — many games run
  in parallel with batched GPU inference. Built into a `selfplay` executable.
- **Python** handles *neural-network training* (PyTorch on GPU), checkpointing,
  benchmarking, and a **PySide6 GUI** for playing against the trained model.

The two halves communicate through a **flat binary game-data file** and a
**TorchScript-exported model**. Each training episode: Python exports the current
model → C++ plays self-play games → Python loads the data and trains → repeat.

> Status: work in progress (per README). Code is mature and heavily optimized,
> but there are no trained weights committed and several environment assumptions
> (see Gotchas) are Windows-centric.

## Repository layout

### Python (training, GUI, reference implementation)

| File | Role |
|------|------|
| [main.py](main.py) | **Two purposes.** (1) The `run_training_parallel_hybrid()` training loop (the active `__main__` block at the bottom). (2) A standalone **PySide6 GUI** (`MainWindow`, `AIWorker`, board widget, eval bar) for a human to play the trained model. |
| [Agent.py](Agent.py) | Core Python **MCTS + Node** classes, the `MCTS.train_network()` gradient step, `evaluate_vs_model()` / `evaluate_vs_random()` benchmarking, the hybrid training orchestration, replay-window/aux-loss schedules, and `run_benchmark()`. This is the heart of the Python side. |
| [ChessEnv.py](ChessEnv.py) | **Reference chess environment.** State encoding (`encode_state` → 20×8×8), action decoding (`apply_action`), legal-move mask (`get_action_mask`), and a hand-written tapered **PeSTO evaluation** (`get_evaluation`). The C++ `ChessEnv` must match this byte-for-byte. |
| [Neuralnet.py](Neuralnet.py) | `CNNNet`: initial conv + 10 residual blocks (128 channels) → **three heads**: policy (4672), value (tanh scalar), and an **auxiliary heuristic head** (tanh scalar). |
| [utils.py](utils.py) | `PrioritizedReplayBuffer` (PER) backed by a binary `_SumTree`. Sparse policy storage, dynamic capacity `grow()`, beta annealing, tree snapshot/restore for checkpointing. |
| [data_loader.py](data_loader.py) | `load_binary_game_data()` — reads the C++ self-play `.bin` file into numpy arrays for the replay buffer. **Defines the binary format contract.** |
| [MCTS_simple.py](MCTS_simple.py) | Standalone classical MCTS (UCB + random rollouts, no NN). Reference / baseline only; not part of the training pipeline. Note: it calls `ChessEnv.get_potential()` which no longer exists (`get_evaluation` is the current name) — treat as stale. |

### C++ (`src/`, parallel self-play engine)

| File | Role |
|------|------|
| [src/main.cpp](src/main.cpp) | CLI entry point for the `selfplay` binary. Parses args, picks sequential `SelfPlay` (`--workers 1`) or `ParallelSelfPlay` (`--workers > 1`), runs games, writes binary data. |
| [src/ChessEnv.h](src/ChessEnv.h) / [src/ChessEnv.cpp](src/ChessEnv.cpp) | **C++ mirror of ChessEnv.py** — must produce identical `encode_state`, `get_action_mask`, `apply_action`, and `get_evaluation` outputs. Uses the Disservin chess-library. |
| [src/MCTS.h](src/MCTS.h) / [src/MCTS.cpp](src/MCTS.cpp) | Synchronous batched MCTS (`Node`, virtual loss, `run_simulation_batch`). Used by the single-threaded `SelfPlay` path. |
| [src/SelfPlay.h](src/SelfPlay.h) / [src/SelfPlay.cpp](src/SelfPlay.cpp) | Sequential self-play loop + `write_data()` (the canonical binary serializer). Defines `Sample`, `GameMeta`, `SelfPlayConfig`. |
| [src/Evaluator.h](src/Evaluator.h) | `EvalQueue` — thread-safe MPSC queue of leaf eval requests with a careful shutdown protocol (worker-done counting + partial-batch flush to avoid deadlock). |
| [src/CentralEvaluator.h](src/CentralEvaluator.h) / [.cpp](src/CentralEvaluator.cpp) | The **single GPU thread** that owns the TorchScript model, drains `EvalQueue` in batches, runs one forward pass per batch, fulfils each request's `std::promise`. Only object allowed to call `model_.forward()`. |
| [src/AsyncMCTS.h](src/AsyncMCTS.h) / [.cpp](src/AsyncMCTS.cpp) | Per-worker MCTS that submits leaves to `EvalQueue` (via futures) instead of calling the model directly. Pipelines GPU eval of batch N with CPU selection of batch N+1. Injects Dirichlet root noise. |
| [src/ParallelSelfPlay.h](src/ParallelSelfPlay.h) / [.cpp](src/ParallelSelfPlay.cpp) | Runs N worker threads, each playing games via `AsyncMCTS`, all sharing one `CentralEvaluator`. Streams samples directly to disk. Holds the **curriculum FENs** (mate-in-1 / overwhelming-material starts for episodes 1–5). |

### Tests (`tests/`)

| File | Validates |
|------|-----------|
| [tests/test_encoding.cpp](tests/test_encoding.cpp) | C++ `encode_state`/`get_action_mask`/`apply_action` against Python golden vectors. |
| [tests/test_mcts.cpp](tests/test_mcts.cpp) | C++ `Node` structure and MCTS simulation (needs a TorchScript model arg). |
| [tests/test_parallel.cpp](tests/test_parallel.cpp) | `EvalQueue`, `AsyncMCTS` with stub evaluator, and full `ParallelSelfPlay`. |
| [tests/test_data_loader.py](tests/test_data_loader.py) | Python `load_binary_game_data()` round-trip. |

The codebase uses a **"Phase N"** vocabulary in comments (Phase 2 = encoding,
Phase 3 = MCTS, Phase 4 = parallel/async, Phase 5 = Python data bridge). This is
historical build-order context, not a runtime concept.

## Key conventions (read before touching ChessEnv or the network)

- **Action space = 4672 = 8×8×73 planes.** Planes: 0–55 queen-like sliding
  (8 directions × 7 distances), 56–63 knight hops, 64–72 under-promotions
  (3 directions × ROOK/BISHOP/KNIGHT). Queen promotion is encoded as a normal
  sliding move, *not* an under-promotion plane.
- **State = 20×8×8 float32.** Channels 0–5 current-player pieces, 6–11 opponent
  pieces, 12 turn (always 1.0), 13–16 castling rights, 17 en-passant, 18
  halfmove-clock/50, 19 repetition-count/2.
- **"Agent always sees itself as White."** When Black is to move, the board is
  flipped vertically and colors swapped. `encode_state`, `apply_action`, and
  `get_action_mask` all operate in this mover-relative frame. Targets `z` and
  `eval_target` are stored mover-relative (negated on Black's plies).
- **Python and C++ ChessEnv MUST stay in sync.** Any change to encoding/action
  decoding/evaluation must be made in *both* and validated with `test_encoding`.
  (Golden vectors come from a `generate_test_vectors.py` referenced in comments
  but not currently committed — regenerate before relying on encoding tests.)
- `num_simulations` **must be divisible by** `leaf_batch_size` (enforced in both
  `main.cpp` and the Python training config).
- The auxiliary head predicts the **mover-relative PeSTO heuristic eval**; it is a
  curriculum signal that is annealed out (see `_get_lambda_aux`).

## Build & run

### C++ self-play binary

Requires **LibTorch** and CMake ≥ 3.18. The chess-library is fetched
automatically via `FetchContent`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Produces: selfplay, test_encoding, test_mcts, test_parallel
```

Run self-play directly (normally invoked by the Python training loop):

```bash
./build/selfplay model_current.pt --games 100 --workers 10 --sims 448 \
    --batch 64 --moves 400 --episode 1 --temp 1.0 --output games_ep0.bin
```

### Python training (the main pipeline)

```bash
pip install torch numpy python-chess pyside6
python main.py    # runs run_training_parallel_hybrid(episodes=500, ...)
```

Each episode: export TorchScript model → spawn the `selfplay` subprocess →
load `.bin` into the PER buffer → train `train_batches` gradient steps →
checkpoint. Resumes automatically from `checkpoint_latest.pt` + `buffer_latest.npz`.

### Play against the model (GUI)

`main.py` contains the GUI but its `__main__` block runs training. To launch the
GUI, call `main()` from `main.py` (or temporarily swap which block runs). It
auto-loads the latest checkpoint (`checkpoint_latest.pt` > `model_ep*.pt`).

### Benchmark

`run_benchmark()` in [Agent.py](Agent.py) plays a current model vs an older one
(or vs a random mover). Its `__main__` guard is intentionally misspelled
(`"__main__a"`) so it stays dormant.

## Generated / runtime artifacts (not committed)

- `checkpoint_latest.pt` — full training state (model, optimizer, scheduler, scaler, episode).
- `buffer_latest.npz` — compressed replay buffer + PER tree snapshot.
- `model_ep{N}.pt` — milestone weights every 10 episodes.
- `model_current.pt` — TorchScript export handed to the C++ subprocess each episode.
- `games_ep{N}.bin` — per-episode C++ self-play output (deleted after load unless `keep_game_files`).
- `training_log.csv` — per-episode metrics (losses, outcomes, buffer %, LR, timing).

## Training schedules (in Agent.py)

- **Replay window** grows 500K → 1.5M over episodes (`_per_target_capacity`).
- **Sims/episode** ramp 100→200→300→full (`_SIM_SCHEDULE`); **games/episode**
  ramp 400→400→200→target (`_GAMES_SCHEDULE`) — AlphaZero-style curriculum.
- **PER beta** anneals 0.4→1.0 across the run; **aux-loss λ** stays 1.0 for
  episodes 1–50, decays to 0 by episode 100, then off.
- C++ **curriculum FENs** (mate-in-1 + huge material edges) replace the start
  position for episodes 1–5 so the value head sees decisive results immediately.

## Gotchas

- **Platform mismatch.** [CMakeLists.txt](CMakeLists.txt) hardcodes
  `Torch_DIR = C:/libtorch/...` and the Python loop defaults
  `selfplay_exe="build/Release/selfplay.exe"` — both Windows paths, while the dev
  machine is macOS. Expect to adjust `Torch_DIR` and the executable path for the
  local platform (e.g. `build/selfplay`).
- C++ self-play prints verbose `[DEBUG worker ...]` lines to stderr; this is
  inherited by the Python subprocess and shown in the terminal.
- `ParallelSelfPlay::run()` calls `torch::set_num_threads(1)` to avoid an OpenMP
  deadlock when many worker threads contend on the model — don't remove it.
- `MCTS_simple.py` references a removed `get_potential()` method — it is stale and
  decoupled from the active pipeline.
- The `EvalQueue` shutdown protocol is subtle (worker-done counting + partial
  batch flush). `min_batch` must be ≤ `leaf_batch_size` or the evaluator can
  deadlock against the last blocked worker.
