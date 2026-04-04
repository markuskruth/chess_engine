import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ChessEnv import ChessEnv
from Neuralnet import CNNNet
from utils import ReplayBuffer
import chess
from collections import deque
import multiprocessing as mp
import os
import time
import threading


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
            node.N[action] += 1
            node.W[action] += value
            node.Q[action] = node.W[action] / node.N[action]
            node.total_N += 1  # used in UCB numerator

            value = -value  # switch perspective


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

        # ── 1. SELECTION (with virtual loss) ────────────────────────────
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

        # ── 2. BATCHED EVALUATION (Fix #5) ──────────────────────────────
        # Write into pre-allocated buffer then take a zero-copy torch view
        for i, leaf in enumerate(leaves):
            self._leaf_batch_buf[i] = leaf.state
        states_t = torch.from_numpy(self._leaf_batch_buf)
        with torch.no_grad():
            p_batch, v_batch = self.model(states_t)

        p_batch = p_batch.cpu().numpy().reshape(self.leaf_batch_size, -1)  # (B, 4672)
        v_batch = v_batch.cpu().numpy().flatten()                           # (B,)

        # ── 3. EXPAND + UNDO VIRTUAL LOSS + BACKPROP ────────────────────
        for i, (leaf, path) in enumerate(zip(leaves, paths)):
            # Guard: two paths in the same batch can land on the same new node
            if not leaf.is_expanded:
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

            value = float(v_batch[i])

            # Correct virtual loss and apply real value in one step:
            # W was decremented by 1; adding (value + 1) cancels it and adds v.
            # N and total_N were already incremented by the virtual loss above.
            for node, action in reversed(path):
                node.N[action]  += 1
                node.W[action]  += value + 1.0
                node.Q[action]   = node.W[action] / node.N[action]
                node.total_N    += 1
                value = -value


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
            # Move limit reached: use evaluation function as terminal signal.
            # get_potential() returns a value in [-1, 1] from White's perspective,
            # so it directly encodes who is winning and by how much.
            z = ChessEnv.get_potential(board)

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

    def get_game_result(self, board):
        """Get result of game."""
        if not board.is_game_over():
            return 0
        result = board.result()
        return 10 if result == "1-0" else (-10 if result == "0-1" else 0)


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

    # Wait for the last episode's buffer save to finish before exiting
    if save_thread is not None:
        save_thread.join()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    # Required on Windows: multiprocessing needs the 'spawn' guard
    mp.freeze_support()

    episodes = 500
    num_workers = 8          # 8 CPU workers; leaves 2 threads for OS + main process
    games_per_worker = 3     # 24 games/episode
    batch_size = 256
    num_simulations = 400    # must be divisible by leaf_batch_size
    leaf_batch_size = 8      # forward passes per MCTS step: 400/8 = 50 batched calls
    max_moves = 200
    train_batches = 100

    run_training_parallel(
        episodes=episodes,
        num_workers=num_workers,
        games_per_worker=games_per_worker,
        batch_size=batch_size,
        num_simulations=num_simulations,
        max_moves=max_moves,
        train_batches=train_batches,
        leaf_batch_size=leaf_batch_size,
    )