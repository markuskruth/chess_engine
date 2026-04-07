import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ChessEnv import ChessEnv
from Neuralnet import CNNNet
from utils import ReplayBuffer
from data_loader import load_binary_game_data
import chess
import csv
import random
import subprocess
import sys
from collections import deque
import multiprocessing as mp
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
        self.memory = ReplayBuffer(capacity=buffer_size)
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


    def play_game(self, temperature=1.0, max_moves=200):
        board = ChessEnv.reset()
        trajectory = []

        for _ in range(max_moves):
            if board.is_game_over():
                break

            state = ChessEnv.encode_state(board)
            turn = board.turn

            # Fresh MCTS tree for each position
            root = Node(state, board.copy())
            num_batched = self.num_simulations // self.leaf_batch_size
            for _ in range(num_batched):
                self.run_simulation_batch(root)

            # Policy from visit counts
            N = root.N.copy().astype(np.float32)
            pi = N / (np.sum(N) + 1e-10)
            action_idx = np.random.choice(len(pi), p=pi)

            trajectory.append({"state": state, "pi": pi, "turn": turn})
            ChessEnv.apply_action(action_idx, board)

        # --- Determine terminal value from White's perspective ---
        if board.is_game_over():
            # Actual game result: +1 White wins, -1 Black wins, 0 draw
            result = board.result()
            if result == "1-0":
                z = 1.0
            elif result == "0-1":
                z = -1.0
            else:
                z = 0.0
        else:
            # Move limit reached: evaluate position and threshold to decisive/draw.
            # Only clearly winning/losing positions (|eval| > 0.3 ≈ +230cp) count as decisive.
            # This gives clean win/draw/loss labels without mimicking the heuristic's
            # continuous scale.
            # Potential is always given from real whites perspective
            pot = ChessEnv.get_evaluation(board)
            if pot > 0.3:
                z = 1.0
            elif pot < -0.3:
                z = -1.0
            else:
                z = 0.0

        # Propagate z to every position, flipping sign for Black positions so
        # the target is always from the current player's perspective (matching
        # the flipped board encoding).
        data = [
            (entry["state"], entry["pi"], z if entry["turn"] == chess.WHITE else -z)
            for entry in trajectory
        ]

        if board.is_game_over():
            result = board.result()
            if result == "1-0":   outcome = "white"
            elif result == "0-1": outcome = "black"
            else:                 outcome = "draw"
        else:
            outcome = "limit"

        meta = {"moves": len(trajectory), "outcome": outcome}
        return data, meta

    def train_network(self, num_batches=50):
        total_vl = total_pl = 0.0
        for _ in range(num_batches):
            batch = self.memory.sample(batch_size=self.batch_size)
            states, target_pis, target_vs = batch

            # Zero-copy numpy→tensor path; non_blocking overlaps GPU transfer (Fix #7)
            states_tensor     = torch.from_numpy(states).to(self.device, non_blocking=True)
            target_pis_tensor = torch.from_numpy(target_pis).to(self.device, non_blocking=True)
            target_vs_tensor  = torch.from_numpy(target_vs).to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type):
                pred_p, pred_v = self.model(states_tensor)
                pred_p = pred_p.view(pred_p.size(0), -1)
                value_loss = nn.MSELoss()(pred_v, target_vs_tensor)
                log_probs = torch.log_softmax(pred_p, dim=1)
                policy_loss = -(target_pis_tensor * log_probs).sum(dim=1).mean()
                loss = value_loss + policy_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_vl += value_loss.item()
            total_pl += policy_loss.item()

        return {
            "value_loss":  total_vl / num_batches,
            "policy_loss": total_pl / num_batches,
            "total_loss":  (total_vl + total_pl) / num_batches,
        }

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


def _save_buffer_bg(states, policies, values, index, size, path):
    """Background thread target: compress and write the replay buffer (Fix #6)."""
    np.savez_compressed(
        path,
        states=states,
        policies=policies,
        values=values,
        index=np.array(index),
        size=np.array(size),
    )


def _worker_play_games(args):
    """
    Top-level worker function for multiprocessing (must be module-level to be
    picklable on Windows, which uses 'spawn' instead of 'fork').

    Runs entirely on CPU — GPU inference at batch-size 1 has too much
    launch overhead to be faster than CPU for this workload.
    """
    state_dict, num_games, num_simulations, max_moves, leaf_batch_size, _ = args

    # Build model directly on CPU — avoids allocating on GPU then immediately moving back (Fix #3)
    agent = MCTS(num_simulations=num_simulations, leaf_batch_size=leaf_batch_size, device='cpu', buffer_size=1)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()

    all_data = []
    all_meta = []
    for _ in range(num_games):
        data, meta = agent.play_game(max_moves=max_moves)
        all_data.extend(data)
        all_meta.append(meta)
    return all_data, all_meta


def run_training(
    episodes,
    num_games_per_episode,
    batch_size,
    num_simulations
    ):
    """Single-process training loop (kept for reference / quick experiments)."""
    print("Initializing MCTS Chess Training Engine...")
    mcts = MCTS(buffer_size=100000, batch_size=batch_size, num_simulations=num_simulations)

    print(f"Device: {mcts.device}")
    print(f"Running {episodes} episodes with {num_games_per_episode} games per episode")
    print(f"Simulations per position: {num_simulations}")
    print(f"Batch size: {batch_size}\n")

    for ep in range(episodes):
        print(f"Episode {ep+1}/{episodes}")

        for game_idx in range(num_games_per_episode):
            game_data, _ = mcts.play_game()
            for state, policy, value in game_data:
                mcts.memory.add(state, policy, np.array([value]))

            if (game_idx + 1) % 25 == 0:
                print(f"  Games: {game_idx + 1}/{num_games_per_episode} | Buffer: {len(mcts.memory)}")

        if len(mcts.memory) > batch_size:
            print(f"  Training... ", end="", flush=True)
            mcts.train_network(num_batches=50)
            print("done")

        if (ep + 1) % 5 == 0:
            torch.save(mcts.model.state_dict(), f"model_checkpoint_ep{ep+1}.pt")
            print(f"  Saved: model_checkpoint_ep{ep+1}.pt\n")

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


def run_training_parallel(
    episodes,
    num_workers=5,
    games_per_worker=2,
    batch_size=64,
    num_simulations=200,
    max_moves=200,
    train_batches=50,
    leaf_batch_size=8,
):
    """
    Parallel training loop.

    Self-play is distributed across `num_workers` CPU processes.
    Network training runs on GPU (or CPU if unavailable) in the main process.

    Each episode:
      1. Broadcast current model weights (CPU copy) to all workers.
      2. Workers play `games_per_worker` games each in parallel.
      3. Main process collects all game data into the replay buffer.
      4. Main process trains the network for `train_batches` gradient steps.
      5. Repeat.

    Recommended for a 6-core CPU: num_workers=5 (leaves 1 core for main).
    """
    print("Initializing Parallel MCTS Training...")
    mcts = MCTS(buffer_size=100000, batch_size=batch_size, num_simulations=num_simulations, leaf_batch_size=leaf_batch_size)

    # --- RESUME ---
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
                mcts.memory.states[:n] = data["states"]
                mcts.memory.policies[:n] = data["policies"]
                mcts.memory.values[:n] = data["values"]
                mcts.memory.index = int(data["index"])
                mcts.memory.size = n
                print(f"  Replay buffer restored: {n} positions")
            except Exception as e:
                print(f"  WARNING: buffer_latest.npz is corrupt ({e}), starting with empty buffer")

    total_games = num_workers * games_per_worker
    print(f"Device (training): {mcts.device}")
    print(f"Workers: {num_workers}  |  Games/episode: {total_games}")
    print(f"Simulations/position: {num_simulations}  |  Batch size: {batch_size}\n")

    episode_times = []
    save_thread = None  # background buffer-save thread (Fix #6)

    # --- EVAL LOG SETUP ---
    eval_log_path = "eval_log.csv"
    eval_interval = 5  # run evaluation every N episodes
    if not os.path.exists(eval_log_path):
        with open(eval_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "win_rate", "draw_rate", "loss_rate",
                                    "value_loss", "policy_loss"])

    def fmt_time(seconds):
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    with mp.Pool(processes=num_workers) as pool:
        for ep in range(start_ep, episodes):
            ep_start = time.time()
            print(f"\n{'─' * 56}")
            print(f"Episode {ep+1}/{episodes}")

            # --- 1. PARALLEL SELF-PLAY ---
            cpu_state_dict = {k: v.cpu() for k, v in mcts.model.state_dict().items()}
            worker_args = [
                (cpu_state_dict, games_per_worker, num_simulations, max_moves, leaf_batch_size, i)
                for i in range(num_workers)
            ]

            sp_start = time.time()
            all_game_data = []
            all_meta = []
            workers_done = 0
            for game_data, game_meta in pool.imap_unordered(_worker_play_games, worker_args):
                workers_done += 1
                all_game_data.append(game_data)
                all_meta.extend(game_meta)
                print(f"\r  Self-play  [{workers_done}/{num_workers} workers done]", end="", flush=True)
            sp_time = time.time() - sp_start

            # Ensure previous episode's buffer save finished before we write to the buffer
            if save_thread is not None:
                save_thread.join()

            # --- 2. COLLECT DATA ---
            total_positions = 0
            for game_data in all_game_data:
                for state, policy, value in game_data:
                    mcts.memory.add(state, policy, np.array([value]))
                    total_positions += 1

            outcomes = [m["outcome"] for m in all_meta]
            avg_moves = sum(m["moves"] for m in all_meta) / max(len(all_meta), 1)
            n_games = len(all_meta)
            w = outcomes.count("white")
            b = outcomes.count("black")
            d = outcomes.count("draw")
            lim = outcomes.count("limit")

            print(f"\r  Self-play  {n_games} games | {total_positions} positions | "
                  f"avg {avg_moves:.0f} moves | "
                  f"W {w/n_games*100:.0f}% D {d/n_games*100:.0f}% B {b/n_games*100:.0f}%"
                  + (f" limit {lim/n_games*100:.0f}%" if lim else "")
                  + f" | {fmt_time(sp_time)}")

            # --- 3. TRAIN ON GPU ---
            tr_start = time.time()
            losses = {}
            if len(mcts.memory) > batch_size:
                losses = mcts.train_network(num_batches=train_batches)
                tr_time = time.time() - tr_start
                print(f"  Training   {train_batches} batches | "
                      f"val {losses['value_loss']:.4f}  "
                      f"pol {losses['policy_loss']:.4f}  "
                      f"total {losses['total_loss']:.4f} | {fmt_time(tr_time)}")
            else:
                print(f"  Training   skipped (buffer {len(mcts.memory)} < batch {batch_size})")

            # --- 4. LR SCHEDULE ---
            mcts.scheduler.step()
            current_lr = mcts.optimizer.param_groups[0]['lr']

            buf_pct = len(mcts.memory) / mcts.memory.capacity * 100
            print(f"  Buffer     {len(mcts.memory):,} / {mcts.memory.capacity:,} ({buf_pct:.1f}%)  "
                  f"LR {current_lr:.6f}")

            # --- 5. ETA ---
            ep_time = time.time() - ep_start
            episode_times.append(ep_time)
            avg_ep_time = sum(episode_times[-20:]) / len(episode_times[-20:])
            remaining_eps = episodes - (ep + 1)
            eta = avg_ep_time * remaining_eps
            print(f"  Episode time: {fmt_time(ep_time)}  |  ETA: {fmt_time(eta)}")

            # --- 6. CHECKPOINT (every episode) ---
            # Model state is small and fast — save synchronously
            torch.save({
                "episode": ep,
                "model": mcts.model.state_dict(),
                "optimizer": mcts.optimizer.state_dict(),
                "scheduler": mcts.scheduler.state_dict(),
                "scaler": mcts.scaler.state_dict(),
            }, "checkpoint_latest.pt")

            # Buffer save is expensive (up to ~2.4 GB compressed) — run in background
            # so it overlaps with the next episode's self-play phase (Fix #6).
            # The join at the top of the loop ensures it finishes before the buffer
            # is modified again during collect.
            n = mcts.memory.size
            save_thread = threading.Thread(
                target=_save_buffer_bg,
                args=(
                    mcts.memory.states[:n],
                    mcts.memory.policies[:n],
                    mcts.memory.values[:n],
                    mcts.memory.index,
                    n,
                    "buffer_latest.npz",
                ),
                daemon=True,
            )
            save_thread.start()
            print(f"  Checkpoint saved (ep {ep+1})")

            if (ep + 1) % 10 == 0:
                torch.save(mcts.model.state_dict(), f"model_ep{ep+1}.pt")
                print(f"  Milestone: model_ep{ep+1}.pt")

            # --- 7. EVALUATION vs RANDOM ---
            if (ep + 1) % eval_interval == 0:
                ev_start = time.time()
                win_r, draw_r, loss_r = mcts.evaluate_vs_random(num_games=20, num_simulations=100)
                ev_time = time.time() - ev_start
                val_loss = losses.get("value_loss", float("nan")) if len(mcts.memory) > batch_size else float("nan")
                pol_loss = losses.get("policy_loss", float("nan")) if len(mcts.memory) > batch_size else float("nan")
                with open(eval_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([ep + 1, f"{win_r:.3f}", f"{draw_r:.3f}", f"{loss_r:.3f}",
                                            f"{val_loss:.4f}", f"{pol_loss:.4f}"])
                print(f"  Eval vs random  W {win_r*100:.0f}%  D {draw_r*100:.0f}%  L {loss_r*100:.0f}%  | {fmt_time(ev_time)}")

    # Wait for the last episode's buffer save to finish before exiting
    if save_thread is not None:
        save_thread.join()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

def run_training_parallel_hybrid(
    episodes,
    selfplay_exe="build/Release/selfplay.exe",
    num_workers=4,
    games_per_episode=20,
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
        buffer_size=100000,
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
                mcts.memory.states[:n]   = data["states"]
                mcts.memory.policies[:n] = data["policies"]
                mcts.memory.values[:n]   = data["values"]
                mcts.memory.index        = int(data["index"])
                mcts.memory.size         = n
                print(f"  Replay buffer restored: {n} positions")
            except Exception as e:
                print(f"  WARNING: buffer_latest.npz corrupt ({e}), starting empty")

    print(f"Device (training): {mcts.device}")
    print(f"Workers: {num_workers}  |  Games/episode: {games_per_episode}")
    print(f"Simulations/position: {num_simulations}  |  Batch size: {batch_size}\n")

    # ── Eval log ──────────────────────────────────────────────────────────────
    eval_log_path = "eval_log.csv"
    eval_interval = 5
    if not os.path.exists(eval_log_path):
        with open(eval_log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "win_rate", "draw_rate", "loss_rate",
                 "value_loss", "policy_loss"]
            )

    def fmt_time(seconds):
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        if h > 0:   return f"{h}h {m:02d}m {s:02d}s"
        if m > 0:   return f"{m}m {s:02d}s"
        return f"{s}s"

    episode_times = []
    save_thread   = None

    for ep in range(start_ep, episodes):
        ep_start = time.time()
        print(f"\n{'─' * 56}")
        print(f"Episode {ep + 1}/{episodes}")

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

        game_file = f"games_ep{ep}.bin"
        cmd = [
            selfplay_exe,
            model_file,
            "--games",   str(games_per_episode),
            "--workers", str(num_workers),
            "--sims",    str(num_simulations),
            "--batch",   str(leaf_batch_size),
            "--moves",   str(max_moves),
            "--temp",    str(temperature),
            "--output",  game_file,
        ]

        # ── 2. C++ self-play subprocess ───────────────────────────────────────
        game_file = f"games_ep{ep}.bin"
        cmd = [
            selfplay_exe,
            model_file,
            "--games",   str(games_per_episode),
            "--workers", str(num_workers),
            "--sims",    str(num_simulations),
            "--batch",   str(leaf_batch_size),
            "--moves",   str(max_moves),
            "--temp",    str(temperature),
            "--output",  game_file,
        ]

        sp_start = time.time()
        print(f"  Self-play  running C++ ({games_per_episode} games, "
              f"{num_workers} workers)...", end="", flush=True)
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

        # Print C++ summary lines (lines starting with spaces or '──')
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
        tr_start = time.time()
        losses = {}
        if len(mcts.memory) >= batch_size:
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
        save_thread = threading.Thread(
            target=_save_buffer_bg,
            args=(
                mcts.memory.states[:n],
                mcts.memory.policies[:n],
                mcts.memory.values[:n],
                mcts.memory.index,
                n,
                "buffer_latest.npz",
            ),
            daemon=True,
        )
        save_thread.start()
        print(f"  Checkpoint saved (ep {ep + 1})")

        if (ep + 1) % 10 == 0:
            torch.save(mcts.model.state_dict(), f"model_ep{ep + 1}.pt")
            print(f"  Milestone: model_ep{ep + 1}.pt")

        # ── 8. Evaluation vs random ───────────────────────────────────────────
        """
        if (ep + 1) % eval_interval == 0:
            ev_start = time.time()
            win_r, draw_r, loss_r = mcts.evaluate_vs_random(
                num_games=20, num_simulations=100
            )
            ev_time  = time.time() - ev_start
            val_loss = losses.get("value_loss",  float("nan"))
            pol_loss = losses.get("policy_loss", float("nan"))
            with open(eval_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ep + 1, f"{win_r:.3f}", f"{draw_r:.3f}", f"{loss_r:.3f}",
                    f"{val_loss:.4f}", f"{pol_loss:.4f}",
                ])
            print(f"  Eval vs random  "
                  f"W {win_r*100:.0f}%  D {draw_r*100:.0f}%  L {loss_r*100:.0f}%"
                  f"  | {fmt_time(ev_time)}")
        """
    if save_thread is not None:
        save_thread.join()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)



def run_benchmark(model_path="checkpoint_latest.pt", num_games=20, num_simulations=100, leaf_batch_size=8):
    if not os.path.exists(model_path):
        print(f"Error: Could not find model file at '{model_path}'")
        return

    print(f"Loading model from {model_path}...")
    
    # Initialize MCTS agent
    # We set buffer_size=1 since we aren't training
    mcts = MCTS(
        num_simulations=num_simulations, 
        leaf_batch_size=leaf_batch_size, 
        buffer_size=1
    )
    
    # Load the checkpoint safely to the configured device
    checkpoint = torch.load(model_path, map_location=mcts.device)
    
    # Handle both your saved formats: dictionary (checkpoint_latest) or raw weights (model_epX)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        mcts.model.load_state_dict(checkpoint["model"])
        print(f"Loaded full checkpoint (from episode {checkpoint.get('episode', 'unknown')}).")
    else:
        mcts.model.load_state_dict(checkpoint)
        print("Loaded raw model weights.")
        
    mcts.model.eval()
    
    print(f"\nStarting benchmark: {num_games} games vs Random Mover...")
    print(f"Agent plays White on even games, Black on odd games.")
    print("Evaluating...\n")
    
    start_time = time.time()
    
    # Call your existing evaluation function
    win_r, draw_r, loss_r = mcts.evaluate_vs_random(
        num_games=num_games, 
        num_simulations=num_simulations
    )
    
    elapsed = time.time() - start_time
    
    print("=" * 40)
    print(" 🎯 BENCHMARK RESULTS")
    print("=" * 40)
    print(f"  Wins:   {win_r * 100:>5.1f}%")
    print(f"  Draws:  {draw_r * 100:>5.1f}%")
    print(f"  Losses: {loss_r * 100:>5.1f}%")
    print("-" * 40)
    print(f"  Time:   {elapsed:.1f} seconds")
    print("=" * 40)
"""
if __name__ == "__main__":
    # You can change the model_path to point to whichever .pt file you want to test
    run_benchmark(
        model_path="checkpoint_latest.pt", 
        num_games=20,           # Increase for a more statistically significant test
        num_simulations=100,    # Number of MCTS rollouts per move
        leaf_batch_size=8       # Make sure this matches how the MCTS was initialized
    )
"""
if __name__ == "__main__":
    # Required on Windows: multiprocessing needs the 'spawn' guard
    mp.freeze_support()

    # ── Hybrid training (C++ self-play + Python training) ─────────────────────
    # Requires a compiled selfplay.exe.  Build with:
    #   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
    run_training_parallel_hybrid(
        episodes=500,
        selfplay_exe="build/Release/selfplay.exe",
        num_workers=10,           # parallel game threads in C++ (--workers) (n_of_cpu_threads - 2)
        games_per_episode=100,    # total games per episode
        batch_size=256,
        num_simulations=448,     # must be divisible by leaf_batch_size
        leaf_batch_size=64,      # larger batch → bigger GPU batches → higher utilization
        max_moves=200,
        train_batches=100,
        temperature=1.0,
        keep_game_files=False,   # delete .bin files after loading
    )

    # ── Pure-Python fallback (uncomment to use instead) ───────────────────────
    # run_training_parallel(
    #     episodes=500,
    #     num_workers=8,
    #     games_per_worker=3,
    #     batch_size=256,
    #     num_simulations=400,
    #     leaf_batch_size=8,
    #     max_moves=200,
    #     train_batches=100,
    # )