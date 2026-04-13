import torch
import torch.optim as optim
import numpy as np
from ChessEnv import ChessEnv
from Neuralnet import CNNNet
from utils import PrioritizedReplayBuffer
from data_loader import load_binary_game_data
import chess
import csv
import random
import subprocess
import sys
from datetime import datetime
import os
import time
import threading
import gc


class Node:
    """Optimized Node with memory efficiency."""
    __slots__ = ['state', 'board', 'P', 'N', 'W', 'Q', 'children', 'is_expanded', 'total_N', 'legal_moves_cache']
    
    def __init__(self, state=None, board=None):
        self.state = state
        self.board = board  # Keep board to avoid encoding/decoding
        self.P = None
        self.N = np.zeros(4672, dtype=np.float32)
        self.W = np.zeros(4672, dtype=np.float32)
        self.Q = np.zeros(4672, dtype=np.float32)
        self.children = {}
        self.is_expanded = False
        self.total_N = 0  # Cache total visits to avoid repeated summing
        self.legal_moves_cache = None


class MCTS:
    def __init__(
        self,
        env: ChessEnv = None,
        buffer_size=100000,
        batch_size=16,
        num_simulations=100,
        leaf_batch_size=8,
        device=None):

        self.env = env if env is not None else ChessEnv()
        self.c = 1.0
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.leaf_batch_size = leaf_batch_size
        # Pre-allocated buffer for batched leaf evaluation — reused every call
        self._leaf_batch_buf = np.zeros((leaf_batch_size, 20, 8, 8), dtype=np.float32)





    def run_simulation(self, root):
        """Single MCTS simulation — used by the game UI."""
        node = root
        path = []

        # Selection phase
        while node.is_expanded:
            ucb = node.Q + self.c * node.P * np.sqrt(node.total_N + 1) / (1 + node.N)
            ucb = np.where(node.legal_moves_cache, ucb, -np.inf)  # cached mask

            action = int(np.argmax(ucb))
            path.append((node, action))

            if action not in node.children:
                # New child: copy only the direct parent's board
                board = node.board.copy()
                valid, _ = ChessEnv.apply_action(action, board)
                if valid:
                    state = ChessEnv.encode_state(board)
                    new_node = Node(state, board)  # board already is the child's (Fix #2)
                    node.children[action] = new_node
                    node = new_node
                break
            else:
                # Existing child: follow the pointer, no board work needed (Fix #2)
                node = node.children[action]

        # Expansion & Evaluation
        value = self.expand(node)

        # Backpropagation
        self.backpropagate(path, value)


    def expand(self, node):
        """Expansion: evaluate node with NN, compute and cache the action mask (Fix #1)."""
        # Terminal nodes have an exact value — no NN call needed or wanted.
        # board.turn is the player to move, who has no legal moves in a terminal state.
        # Checkmate: that player is mated → v = -1 from their perspective.
        # Any draw: v = 0.
        if node.board.is_game_over():
            v = -1.0 if node.board.is_checkmate() else 0.0
            node.P = np.zeros(4672, dtype=np.float32)
            node.legal_moves_cache = np.zeros(4672, dtype=bool)
            node.is_expanded = True
            return v

        state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, v = self.model(state_tensor)

        p = p.cpu().numpy().flatten()
        v = v.cpu().numpy().item()

        # Compute mask from the stored board — no decode_state needed (Fix #1)
        mask = ChessEnv.get_action_mask(board=node.board).reshape(-1)
        node.legal_moves_cache = mask  # cache for all future selections through this node

        p = np.where(mask, p, -np.inf)
        valid = ~np.isinf(p)
        if np.any(valid):
            p_max = np.max(p[valid])
            p = np.where(valid, np.exp(p - p_max), 0.0)
        else:
            p = np.zeros_like(p)
        p /= (np.sum(p) + 1e-10)

        node.P = p
        node.is_expanded = True
        return v


    def backpropagate(self, path, value):
        """Backpropagation phase: update statistics along the path."""
        for node, action in reversed(path):
            value = -value  # switch perspective
            node.N[action] += 1
            node.W[action] += value
            node.Q[action] = node.W[action] / node.N[action]
            node.total_N += 1  # used in UCB numerator


    def run_simulation_batch(self, root):
        """
        Run `leaf_batch_size` simulations in one batched forward pass.

        Optimisations applied vs. the naive per-simulation loop:
          - Selection uses the cached action mask on each node — no decode_state (Fix #1)
          - Traversing existing children needs no board copy or move application (Fix #2)
          - New children copy only the direct parent's stored board (Fix #2)
          - Leaf states are written into a pre-allocated buffer; torch.from_numpy
            gives a zero-copy view for the forward pass (Fix #5)
          - Virtual loss (W -= 1) during selection diversifies paths within the batch
        """
        paths  = []
        leaves = []

        # SELECTION (with virtual loss)
        for _ in range(self.leaf_batch_size):
            node = root
            path = []

            while node.is_expanded:
                ucb = (node.Q
                       + self.c * node.P * np.sqrt(node.total_N + 1)
                       / (1 + node.N))
                ucb = np.where(node.legal_moves_cache, ucb, -np.inf)  # Fix #1

                action = int(np.argmax(ucb))
                path.append((node, action))

                # Virtual loss: make this edge look worse to later sims in the batch.
                # Clamp denominator to 1 so the first visit (N=0) doesn't divide by zero.
                node.W[action] -= 1.0
                node.Q[action]  = node.W[action] / max(node.N[action], 1.0)

                if action not in node.children:
                    # New child: copy only the parent's stored board (Fix #2)
                    board = node.board.copy()
                    valid, _ = ChessEnv.apply_action(action, board)
                    if valid:
                        state = ChessEnv.encode_state(board)
                        new_node = Node(state, board)  # board is already the child's (Fix #2)
                        node.children[action] = new_node
                        node = new_node
                    break
                else:
                    # Existing child: follow pointer, no board work at all (Fix #2)
                    node = node.children[action]

            paths.append(path)
            leaves.append(node)

        # BATCHED EVALUATION
        # Write into pre-allocated buffer then take a zero-copy torch view
        for i, leaf in enumerate(leaves):
            self._leaf_batch_buf[i] = leaf.state
        states_t = torch.from_numpy(self._leaf_batch_buf).to(self.device, non_blocking=True)
        with torch.no_grad():
            p_batch, v_batch = self.model(states_t)

        p_batch = p_batch.cpu().numpy().reshape(self.leaf_batch_size, -1)  # (B, 4672)
        v_batch = v_batch.cpu().numpy().flatten()                           # (B,)

        # EXPAND + UNDO VIRTUAL LOSS + BACKPROP
        for i, (leaf, path) in enumerate(zip(leaves, paths)):
            # Guard: two paths in the same batch can land on the same new node
            if not leaf.is_expanded:
                if leaf.board.is_game_over():
                    # Exact terminal value — never use the NN output for this
                    value = -1.0 if leaf.board.is_checkmate() else 0.0
                    leaf.P = np.zeros(4672, dtype=np.float32)
                    leaf.legal_moves_cache = np.zeros(4672, dtype=bool)
                    leaf.is_expanded = True
                else:
                    value = float(v_batch[i])
                    p = p_batch[i]
                    # Compute mask from stored board — no decode_state (Fix #1)
                    mask = ChessEnv.get_action_mask(board=leaf.board).reshape(-1)
                    leaf.legal_moves_cache = mask
                    p = np.where(mask, p, -np.inf)
                    valid = ~np.isinf(p)
                    if np.any(valid):
                        p_max = np.max(p[valid])
                        p = np.where(valid, np.exp(p - p_max), 0.0)
                    else:
                        p = np.zeros_like(p)
                    leaf.P = p / (np.sum(p) + 1e-10)
                    leaf.is_expanded = True
            else:
                # Already expanded (duplicate path in same batch): get exact value
                # for terminals, NN value for non-terminals
                value = (-1.0 if leaf.board.is_checkmate() else 0.0) \
                        if leaf.board.is_game_over() else float(v_batch[i])

            # Correct virtual loss and apply real value in one step:
            # W was decremented by 1; adding (value + 1) cancels it and adds v.
            # N and total_N were already incremented by the virtual loss above.
            for node, action in reversed(path):
                value = -value  # flip perspective
                node.N[action]  += 1
                node.W[action]  += value + 1.0
                node.Q[action]   = node.W[action] / node.N[action]
                node.total_N    += 1


    def train_network(self, num_batches):
        total_vl = total_pl = 0.0
        for _ in range(num_batches):
            states, target_pis, target_vs, indices, is_weights = \
                self.memory.sample(batch_size=self.batch_size)

            # Zero-copy numpy→tensor path; non_blocking overlaps GPU transfer (Fix #7)
            states_tensor     = torch.from_numpy(states).to(self.device, non_blocking=True)
            target_pis_tensor = torch.from_numpy(target_pis).to(self.device, non_blocking=True)
            target_vs_tensor  = torch.from_numpy(target_vs).to(self.device, non_blocking=True)
            is_weights_tensor = torch.from_numpy(is_weights).to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type):
                pred_p, pred_v = self.model(states_tensor)
                pred_p = pred_p.view(pred_p.size(0), -1)

                # Per-sample losses (shape: [batch])
                value_loss_per  = (pred_v - target_vs_tensor).pow(2).squeeze(1)
                log_probs       = torch.log_softmax(pred_p, dim=1)
                policy_loss_per = -(target_pis_tensor * log_probs).sum(dim=1)

                # IS-weighted mean — corrects for non-uniform sampling bias
                loss = (is_weights_tensor * (value_loss_per + policy_loss_per)).mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update priorities using absolute value-head error (analogous to TD error)
            with torch.no_grad():
                td_errors = value_loss_per.detach().float().sqrt().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

            total_vl += value_loss_per.mean().item()
            total_pl += policy_loss_per.mean().item()

        return {
            "value_loss":  total_vl / num_batches,
            "policy_loss": total_pl / num_batches,
            "total_loss":  (total_vl + total_pl) / num_batches,
        }

    def evaluate_vs_model(self, opponent_mcts, num_games=20, num_simulations=100):
        """
        Play num_games against another MCTS agent (alternating colors).
        self is the 'current' model; opponent_mcts is the 'old' model.
        Returns (win_rate, draw_rate, loss_rate) from self's perspective.
        """
        self.model.eval()
        opponent_mcts.model.eval()
        wins = draws = losses = 0

        def pick_action(agent, board):
            state = ChessEnv.encode_state(board)
            root = Node(state, board.copy())
            num_batched = num_simulations // agent.leaf_batch_size
            for _ in range(num_batched):
                agent.run_simulation_batch(root)
            return int(np.argmax(root.N))

        for i in range(num_games):
            board = chess.Board(ChessEnv.reset().fen())
            current_color = chess.WHITE if i % 2 == 0 else chess.BLACK

            while not board.is_game_over() and board.fullmove_number < 100:
                if board.turn == current_color:
                    action_idx = pick_action(self, board)
                else:
                    action_idx = pick_action(opponent_mcts, board)
                ChessEnv.apply_action(action_idx, board)

            result = board.result()
            if (result == "1-0" and current_color == chess.WHITE) or \
               (result == "0-1" and current_color == chess.BLACK):
                wins += 1
            elif (result == "1-0" and current_color == chess.BLACK) or \
                 (result == "0-1" and current_color == chess.WHITE):
                losses += 1
            elif result == "*":
                pot = ChessEnv.get_evaluation(board)
                if (pot > 0.5 and current_color == chess.WHITE) or \
                   (pot < -0.5 and current_color == chess.BLACK):
                    wins += 1
                elif (pot < -0.5 and current_color == chess.WHITE) or \
                     (pot > 0.5 and current_color == chess.BLACK):
                    losses += 1
                else:
                    draws += 1
            else:
                draws += 1

            print(f"  Game {i+1}/{num_games}: {result}  (current={'White' if current_color == chess.WHITE else 'Black'})")

        return wins / num_games, draws / num_games, losses / num_games

    def evaluate_vs_random(self, num_games=20, num_simulations=100):
        """
        Play num_games against a random mover (alternating colors).
        Returns (win_rate, draw_rate, loss_rate) from the model's perspective.
        """
        self.model.eval()
        wins = draws = losses = 0

        for i in range(num_games):
            board = chess.Board(ChessEnv.reset().fen())
            model_color = chess.WHITE if i % 2 == 0 else chess.BLACK

            while not board.is_game_over() and board.fullmove_number < 100:
                if board.turn == model_color:
                    state = ChessEnv.encode_state(board)
                    root = Node(state, board.copy())
                    num_batched = num_simulations // self.leaf_batch_size
                    for _ in range(num_batched):
                        self.run_simulation_batch(root)
                    action_idx = int(np.argmax(root.N))
                    ChessEnv.apply_action(action_idx, board)
                else:
                    board.push(random.choice(list(board.legal_moves)))

            result = board.result()
            if (result == "1-0" and model_color == chess.WHITE) or \
               (result == "0-1" and model_color == chess.BLACK):
                wins += 1
            elif (result == "1-0" and model_color == chess.BLACK) or \
                 (result == "0-1" and model_color == chess.WHITE):
                losses += 1
            elif result == "*":
                # Move limit: use same threshold logic as training
                pot = ChessEnv.get_evaluation(board)
                if (pot > 0.5 and model_color == chess.WHITE) or \
                   (pot < -0.5 and model_color == chess.BLACK):
                    wins += 1
                elif (pot < -0.5 and model_color == chess.WHITE) or \
                     (pot > 0.5 and model_color == chess.BLACK):
                    losses += 1
                else:
                    draws += 1
            else:
                draws += 1

        return wins / num_games, draws / num_games, losses / num_games


# ── Dynamic Replay Window schedule ────────────────────────────────────────────

_PER_INITIAL_CAP = 500_000
_PER_CAP_STEP    = 250_000
_PER_MAX_CAP     = 1_500_000
_PER_GROW_EVERY  = 10   # episodes between each growth step

def _per_target_capacity(ep_0indexed):
    """Return the buffer capacity for the given 0-indexed episode number.

    Schedule (0-indexed episode → capacity):
      0–9   → 500 K
      10–19 → 750 K
      20–29 → 1 M
      ...
      60+   → 2 M  (capped)
    """
    steps = ep_0indexed // _PER_GROW_EVERY
    return min(_PER_INITIAL_CAP + steps * _PER_CAP_STEP, _PER_MAX_CAP)


def _save_buffer_bg(states, policy_indices, policy_values, values, index, size, path, tree):
    """Background thread target: compress and write the replay buffer."""
    np.savez_compressed(
        path,
        states=states,
        policy_indices=policy_indices,
        policy_values=policy_values,
        values=values,
        index=np.array(index),
        size=np.array(size),
        tree=tree,
    )


def run_training_parallel_hybrid(
    episodes,
    selfplay_exe="build/Release/selfplay.exe",
    num_workers=10,
    games_per_episode=40,
    batch_size=256,
    num_simulations=400,
    leaf_batch_size=8,
    max_moves=200,
    train_batches=100,
    temperature=1.0,
    keep_game_files=False,
):
    """
    Hybrid training loop: C++ handles self-play, Python handles training.

    Each episode:
      1. Export current model weights as a TorchScript file.
      2. Invoke the C++ selfplay binary as a subprocess (blocking).
      3. Load the binary game data into the replay buffer.
      4. Train the network on GPU for train_batches gradient steps.
      5. Checkpoint and repeat.

    Parameters
    ----------
    selfplay_exe   : path to the compiled selfplay executable
    num_workers    : parallel game threads passed to C++ (--workers)
    games_per_episode : total games per episode
    keep_game_files   : if False, delete the .bin file after loading
    """
    print("Initializing Hybrid C++/Python Training...")
    mcts = MCTS(
        buffer_size=_PER_INITIAL_CAP,
        batch_size=batch_size,
        num_simulations=num_simulations,
        leaf_batch_size=leaf_batch_size,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_ep = 0
    if os.path.exists("checkpoint_latest.pt"):
        print("Found checkpoint_latest.pt — resuming...")
        ckpt = torch.load("checkpoint_latest.pt", map_location=mcts.device)
        mcts.model.load_state_dict(ckpt["model"])
        mcts.optimizer.load_state_dict(ckpt["optimizer"])
        mcts.scheduler.load_state_dict(ckpt["scheduler"])
        mcts.scaler.load_state_dict(ckpt["scaler"])
        start_ep = ckpt["episode"] + 1
        print(f"  Resumed from episode {ckpt['episode'] + 1}")

        if os.path.exists("buffer_latest.npz"):
            try:
                data = np.load("buffer_latest.npz")
                n = int(data["size"])
                # Grow to accommodate resumed episode's target capacity and any existing data
                resume_cap = max(_per_target_capacity(start_ep), n)
                resume_cap = min(resume_cap, _PER_MAX_CAP)
                if resume_cap > mcts.memory.capacity:
                    mcts.memory.grow(resume_cap)
                mcts.memory.states[:n] = data["states"]
                if "policy_indices" in data:
                    mcts.memory.policy_indices[:n] = data["policy_indices"]
                    mcts.memory.policy_values[:n]  = data["policy_values"]
                else:
                    print("  WARNING: old dense-policy checkpoint — policies will be empty")
                mcts.memory.values[:n] = data["values"]
                mcts.memory.index        = int(data["index"])
                mcts.memory.size         = n
                if "tree" in data:
                    mcts.memory.restore_tree(data["tree"])
                else:
                    mcts.memory.restore_uniform()
                print(f"  Replay buffer restored: {n} positions")
            except Exception as e:
                print(f"  WARNING: buffer_latest.npz corrupt ({e}), starting empty")

    print(f"Device (training): {mcts.device}")
    print(f"Workers: {num_workers}  |  Games/episode: scheduled (400→200→{games_per_episode})")
    print(f"Simulations/position: scheduled (64→128→192→{num_simulations})  |  Batch size: {batch_size}\n")

    # ── Training log ──────────────────────────────────────────────────────────
    train_log_path = "training_log.csv"
    _TRAIN_LOG_HEADER = [
        "timestamp", "episode", "games", "sims", "positions",
        "selfplay_s", "train_s", "episode_s",
        "value_loss", "policy_loss", "total_loss",
        "buffer_size", "buffer_pct", "lr",
        "white_pct", "black_pct", "draw_pct", "limit_pct", "avg_moves",
    ]
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w", newline="") as f:
            csv.writer(f).writerow(_TRAIN_LOG_HEADER)

    def _parse_selfplay_summary(stdout_text):
        """Extract game outcome stats from the C++ selfplay summary block."""
        stats = {
            "white_pct": float("nan"), "black_pct": float("nan"),
            "draw_pct":  float("nan"), "limit_pct": float("nan"),
            "avg_moves": float("nan"),
        }
        for line in stdout_text.splitlines():
            try:
                if "Avg moves" in line:
                    stats["avg_moves"] = float(line.split(":")[1].strip())
                elif "White wins" in line:
                    stats["white_pct"] = float(line.split("(")[1].split("%")[0].strip())
                elif "Black wins" in line:
                    stats["black_pct"] = float(line.split("(")[1].split("%")[0].strip())
                elif "Draws" in line:
                    stats["draw_pct"] = float(line.split("(")[1].split("%")[0].strip())
                elif "Move limit" in line:
                    stats["limit_pct"] = float(line.split("(")[1].split("%")[0].strip())
            except (IndexError, ValueError):
                pass
        return stats

    def fmt_time(seconds):
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        if h > 0:   return f"{h}h {m:02d}m {s:02d}s"
        if m > 0:   return f"{m}m {s:02d}s"
        return f"{s}s"

    # Incremental deepening schedule (mirrors AlphaZero curriculum).
    # Raw targets are rounded up to the nearest multiple of leaf_batch_size so
    # the C++ runner's divisibility requirement is always satisfied.
    _SIM_SCHEDULE = [
        (50, 50),
        (90, 100),
        (250, 200),
        (float("inf"), num_simulations),  # full depth for the rest of training
    ]

    def get_sims_for_episode(ep_1indexed):
        """Return sim count for this episode, snapped up to a leaf_batch_size multiple.
        Minimum is 2 * leaf_batch_size so num_batches >= 2 and tree search actually runs."""
        for cutoff, target in _SIM_SCHEDULE:
            if ep_1indexed <= cutoff:
                remainder = target % leaf_batch_size
                snapped = target if remainder == 0 else target + (leaf_batch_size - remainder)
                return max(snapped, 4 * leaf_batch_size)
        return num_simulations  # unreachable but safe fallback

    _GAMES_SCHEDULE = [
        (50, 400),
        (100, 400),
        (300, 200),
        (float("inf"), games_per_episode),
    ]

    def get_games_for_episode(ep_1indexed):
        for cutoff, target in _GAMES_SCHEDULE:
            if ep_1indexed <= cutoff:
                return target
        return games_per_episode  # unreachable but safe fallback

    episode_times = []
    save_thread   = None

    for ep in range(start_ep, episodes):
        ep_start = time.time()
        print(f"\n{'─' * 56}")
        print(f"Episode {ep + 1}/{episodes}")

        # ── 0. Dynamic Replay Window ──────────────────────────────────────────
        target_cap = _per_target_capacity(ep)
        if target_cap > mcts.memory.capacity:
            old_cap = mcts.memory.capacity
            mcts.memory.grow(target_cap)
            print(f"  Buffer     grew {old_cap:,} → {target_cap:,}")

        # ── 1. Export TorchScript model ───────────────────────────────────────
        model_file = "model_current.pt"
        # Move model to CPU for a clean, context-free export
        mcts.model.eval()
        cpu_model = mcts.model.to("cpu")
        sample_input = torch.randn(1, 20, 8, 8).to("cpu")
        
        with torch.no_grad():
            traced = torch.jit.trace(cpu_model, sample_input)
        traced.save(model_file)
        
        # Move model back to GPU so Python can use it later
        mcts.model.to(mcts.device)

        # Force Python to release all unused GPU memory BEFORE C++ starts
        gc.collect()
        if mcts.device.type == "cuda":
            torch.cuda.empty_cache()

        # ── 2. C++ self-play subprocess ───────────────────────────────────────
        ep_sims   = get_sims_for_episode(ep + 1)
        ep_games  = get_games_for_episode(ep + 1)
        game_file = f"games_ep{ep}.bin"
        cmd = [
            selfplay_exe,
            model_file,
            "--games",   str(ep_games),
            "--workers", str(num_workers),
            "--sims",    str(ep_sims),
            "--batch",   str(leaf_batch_size),
            "--moves",   str(max_moves),
            "--episode", str(ep + 1),
            "--temp",    str(temperature),
            "--output",  game_file,
        ]

        sp_start = time.time()
        print(f"  Self-play  running C++ ({ep_games} games, "
              f"{num_workers} workers, {ep_sims} sims)...", end="", flush=True)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None,   # inherit: debug output goes directly to terminal
                text=True,
            )
        except FileNotFoundError:
            print(f"\n  [ERROR] selfplay executable not found: {selfplay_exe}")
            print(f"  Build with: cmake --build build --config Release")
            sys.exit(1)
        try:
            stdout_data, _ = proc.communicate()
        except (KeyboardInterrupt, SystemExit):
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        if proc.returncode != 0:
            print(f"\n  [ERROR] selfplay exited with code {proc.returncode}")
            sys.exit(1)
        result = subprocess.CompletedProcess(cmd, proc.returncode, stdout_data, "")
        sp_time = time.time() - sp_start
        print(f"\r  Self-play  done in {fmt_time(sp_time)}")

        # Parse and print C++ summary lines
        sp_stats = _parse_selfplay_summary(result.stdout)
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("──") or stripped.startswith("White") \
                    or stripped.startswith("Black") or stripped.startswith("Draw") \
                    or stripped.startswith("Avg") or stripped.startswith("Games") \
                    or stripped.startswith("Move"):
                print(f"    {stripped}")

        # ── 3. Load binary data into replay buffer ────────────────────────────
        if os.path.exists(game_file):
            # Ensure previous buffer-save thread is done before modifying buffer
            if save_thread is not None:
                save_thread.join()

            states, policies, values = load_binary_game_data(game_file)
            mcts.memory.add_batch(states, policies, values)
            total_positions = len(states)
            print(f"  Loaded     {total_positions} positions from {game_file}")

            if not keep_game_files:
                os.remove(game_file)
        else:
            print(f"  WARNING: {game_file} not found — skipping data load")
            total_positions = 0

        # ── 4. Train on GPU ───────────────────────────────────────────────────
        # Anneal PER beta from 0.4 → 1.0 over the full training run
        mcts.memory.set_beta(0.4 + 0.6 * (ep - start_ep) / max(episodes - start_ep - 1, 1))
        tr_start = time.time()
        losses = {}
        if len(mcts.memory) >= batch_size * train_batches:
            mcts.model.train() # Switch model to train mode
            losses = mcts.train_network(num_batches=train_batches)
            tr_time = time.time() - tr_start
            print(f"  Training   {train_batches} batches | "
                  f"val {losses['value_loss']:.4f}  "
                  f"pol {losses['policy_loss']:.4f}  "
                  f"total {losses['total_loss']:.4f} | {fmt_time(tr_time)}")
        else:
            print(f"  Training   skipped (buffer {len(mcts.memory)} < batch {batch_size})")

        # ── 5. LR schedule ────────────────────────────────────────────────────
        mcts.scheduler.step()
        current_lr = mcts.optimizer.param_groups[0]["lr"]
        buf_pct    = len(mcts.memory) / mcts.memory.capacity * 100
        print(f"  Buffer     {len(mcts.memory):,} / {mcts.memory.capacity:,} "
              f"({buf_pct:.1f}%)  LR {current_lr:.6f}")

        # ── 6. ETA ────────────────────────────────────────────────────────────
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        avg_ep_time    = sum(episode_times[-20:]) / len(episode_times[-20:])
        remaining_eps  = episodes - (ep + 1)
        eta            = avg_ep_time * remaining_eps
        print(f"  Episode time: {fmt_time(ep_time)}  |  ETA: {fmt_time(eta)}")

        # ── 7. Checkpoint ─────────────────────────────────────────────────────
        torch.save({
            "episode":   ep,
            "model":     mcts.model.state_dict(),
            "optimizer": mcts.optimizer.state_dict(),
            "scheduler": mcts.scheduler.state_dict(),
            "scaler":    mcts.scaler.state_dict(),
        }, "checkpoint_latest.pt")

        n = mcts.memory.size
        tree_snap = mcts.memory.get_tree_snapshot()
        save_thread = threading.Thread(
            target=_save_buffer_bg,
            args=(
                mcts.memory.states[:n],
                mcts.memory.policy_indices[:n],
                mcts.memory.policy_values[:n],
                mcts.memory.values[:n],
                mcts.memory.index,
                n,
                "buffer_latest.npz",
                tree_snap,
            ),
        )
        save_thread.start()
        print(f"  Checkpoint saved (ep {ep + 1})")

        if (ep + 1) % 10 == 0:
            torch.save(mcts.model.state_dict(), f"model_ep{ep + 1}.pt")
            print(f"  Milestone: model_ep{ep + 1}.pt")

        # ── 8. Training log ───────────────────────────────────────────────────
        with open(train_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ep + 1,
                ep_games,
                ep_sims,
                total_positions,
                f"{sp_time:.1f}",
                f"{tr_time:.1f}" if losses else "",
                f"{ep_time:.1f}",
                f"{losses.get('value_loss',  float('nan')):.4f}" if losses else "",
                f"{losses.get('policy_loss', float('nan')):.4f}" if losses else "",
                f"{losses.get('total_loss',  float('nan')):.4f}" if losses else "",
                len(mcts.memory),
                f"{buf_pct:.1f}",
                f"{current_lr:.6f}",
                f"{sp_stats['white_pct']:.1f}",
                f"{sp_stats['black_pct']:.1f}",
                f"{sp_stats['draw_pct']:.1f}",
                f"{sp_stats['limit_pct']:.1f}",
                f"{sp_stats['avg_moves']:.1f}",
            ])

    if save_thread is not None:
        save_thread.join()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)



def _load_mcts_from_path(path, num_simulations, leaf_batch_size):
    """Load a model checkpoint into a new MCTS agent (CPU or GPU auto-detected)."""
    mcts = MCTS(num_simulations=num_simulations, leaf_batch_size=leaf_batch_size, buffer_size=1)
    checkpoint = torch.load(path, map_location=mcts.device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        mcts.model.load_state_dict(checkpoint["model"])
        ep = checkpoint.get("episode", "unknown")
        print(f"  Loaded '{path}'  (episode {ep})")
    else:
        mcts.model.load_state_dict(checkpoint)
        print(f"  Loaded '{path}'  (raw weights)")
    mcts.model.eval()
    return mcts


def run_benchmark(current_model_path="checkpoint_latest.pt",
                  old_model_path=None,
                  num_games=20,
                  num_simulations=100,
                  leaf_batch_size=8):
    """
    Benchmark the current model against an older version of itself.
    If old_model_path is None, falls back to playing against a random mover.
    """
    for path in [p for p in [current_model_path, old_model_path] if p]:
        if not os.path.exists(path):
            print(f"Error: Could not find model file at '{path}'")
            return

    print("Loading models...")
    current = _load_mcts_from_path(current_model_path, num_simulations, leaf_batch_size)

    start_time = time.time()

    if old_model_path is not None:
        old = _load_mcts_from_path(old_model_path, num_simulations, leaf_batch_size)
        print(f"\nStarting benchmark: {num_games} games  (current vs old)")
        print(f"Current model plays White on even games, Black on odd games.\n")
        win_r, draw_r, loss_r = current.evaluate_vs_model(
            old, num_games=num_games, num_simulations=num_simulations)
        opponent_label = f"old model ({old_model_path})"
    else:
        print(f"\nStarting benchmark: {num_games} games vs Random Mover...")
        print(f"Agent plays White on even games, Black on odd games.\n")
        win_r, draw_r, loss_r = current.evaluate_vs_random(
            num_games=num_games, num_simulations=num_simulations)
        opponent_label = "random mover"

    elapsed = time.time() - start_time

    print()
    print("=" * 44)
    print(f"  BENCHMARK RESULTS  (vs {opponent_label})")
    print("=" * 44)
    print(f"  Wins:   {win_r * 100:>5.1f}%")
    print(f"  Draws:  {draw_r * 100:>5.1f}%")
    print(f"  Losses: {loss_r * 100:>5.1f}%")
    print("-" * 44)
    print(f"  Time:   {elapsed:.1f} seconds")
    print("=" * 44)


if __name__ == "__main__":
    # You can change the model_path to point to whichever .pt file you want to test
    run_benchmark(
    current_model_path="checkpoint_latest.pt",
    old_model_path="model_ep60.pt",
    num_games=20,
    num_simulations=100,
    leaf_batch_size=8
)

# TODO: Maybe drop learning rate to 0.0001
if __name__ == "__main__z":
    # ── Hybrid training (C++ self-play + Python training) ─────────────────────
    # Requires a compiled selfplay.exe.  Build with:
    #   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
    run_training_parallel_hybrid(
        episodes=500,
        selfplay_exe="build/Release/selfplay.exe",
        num_workers=10,           # parallel game threads in C++ (--workers) (n_of_cpu_threads - 2)
        games_per_episode=100,    # total games per episode
        batch_size=2048,
        num_simulations=448,      # must be divisible by leaf_batch_size
        leaf_batch_size=64,       # larger batch → bigger GPU batches → higher utilization
        max_moves=400,
        train_batches=50,
        temperature=1.0,
        keep_game_files=False,    # delete .bin files after loading
    )