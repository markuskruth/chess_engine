import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ChessEnv import ChessEnv
from Neuralnet import CNNNet
from utils import ReplayBuffer
import chess
from collections import deque


class Node:
    """Optimized Node with memory efficiency."""
    __slots__ = ['state', 'board', 'P', 'N', 'W', 'Q', 'children', 'is_expanded', 'total_N', 'legal_moves_cache']
    
    def __init__(self, state=None, board=None):
        self.state = state
        self.board = board  # Keep board to avoid encoding/decoding
        self.P = None
        self.N = np.zeros(4672, dtype=np.uint32)  # Use uint32 for memory efficiency
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
        num_simulations=100):

        self.env = env if env is not None else ChessEnv()
        self.c = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        
        # Cache for action masks
        self._action_mask_cache = {}

    def get_legal_moves_array(self, board):
        """Get legal moves and convert to proper action indices."""
        return list(board.legal_moves)

    def mask_illegal_moves(self, actions, board=None, state=None):
        """Mask illegal moves using the board directly (faster)."""
        if board is not None:
            action_mask = ChessEnv.get_action_mask(board=board)
        else:
            action_mask = ChessEnv.get_action_mask(state=state)
        action_mask = action_mask.reshape(-1)
        masked_actions = np.where(action_mask, actions, -np.inf)
        return masked_actions



    def run_simulation(self, root):
        """Optimized single MCTS simulation."""
        node = root
        path = []
        board = root.board.copy()  # Copy board once for simulation

        # Selection phase
        while node.is_expanded:
            ucb = node.Q + self.c * node.P * np.sqrt(node.total_N + 1) / (1 + node.N)
            # Use node.state (encoded from current player's perspective) for correct masking
            ucb = self.mask_illegal_moves(ucb, state=node.state)

            action = int(np.argmax(ucb))
            path.append((node, action))

            if action not in node.children:
                # Create new child node by applying the action
                valid, _ = ChessEnv.apply_action(action, board)
                if valid:
                    state = ChessEnv.encode_state(board)
                    new_node = Node(state, board.copy())
                    node.children[action] = new_node
                    node = new_node
                break
            else:
                # Traverse existing child
                valid, _ = ChessEnv.apply_action(action, board)
                if valid:
                    node = node.children[action]
                else:
                    break

        # Expansion & Evaluation
        value = self.expand(node)

        # Backpropagation
        self.backpropagate(path, value)


    def expand(self, node):
        """Expansion phase: evaluate node with neural network."""
        state = node.state

        # Get logprobs and value estimate from neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, v = self.model(state_tensor)

        # Flatten policy from (1, 8, 8, 73) to (4672,)
        p = p.cpu().numpy().flatten()
        v = v.cpu().numpy().item()

        # Mask illegal moves (-inf), then apply masked softmax
        p = self.mask_illegal_moves(p, state=state)
        valid = ~np.isinf(p)
        if np.any(valid):
            p_max = np.max(p[valid])
            p = np.where(valid, np.exp(p - p_max), 0.0)
        else:
            p = np.zeros_like(p)
        p = p / (np.sum(p) + 1e-10)

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
            for _ in range(self.num_simulations):
                self.run_simulation(root)

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
        return data

    def train_network(self, num_batches=50):
        for _ in range(num_batches):
            batch = self.memory.sample(batch_size=self.batch_size)
            states, target_pis, target_vs = batch

            # Batch tensor conversion
            states_tensor = torch.FloatTensor(states).to(self.device)
            target_pis_tensor = torch.FloatTensor(target_pis).to(self.device)
            target_vs_tensor = torch.FloatTensor(target_vs).to(self.device)

            # Forward pass
            pred_p, pred_v = self.model(states_tensor)
            pred_p = pred_p.view(pred_p.size(0), -1)

            # Combined loss
            value_loss = nn.MSELoss()(pred_v, target_vs_tensor)
            log_probs = torch.log_softmax(pred_p, dim=1)
            policy_loss = -(target_pis_tensor * log_probs).sum(dim=1).mean()
            
            loss = value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_game_result(self, board):
        """Get result of game."""
        if not board.is_game_over():
            return 0
        result = board.result()
        return 10 if result == "1-0" else (-10 if result == "0-1" else 0)


def run_training(
    episodes,
    num_games_per_episode,
    batch_size,
    num_simulations
    ):
    """training loop."""
    print("Initializing MCTS Chess Training Engine...")
    mcts = MCTS(buffer_size=100000, batch_size=batch_size, num_simulations=num_simulations)
    
    print(f"Device: {mcts.device}")
    print(f"Running {episodes} episodes with {num_games_per_episode} games per episode")
    print(f"Simulations per position: {num_simulations}")
    print(f"Batch size: {batch_size}\n")

    for ep in range(episodes):
        print(f"Episode {ep+1}/{episodes}")

        # 1. SELF-PLAY
        for game_idx in range(num_games_per_episode):
            game_data = mcts.play_game()
            for state, policy, value in game_data:
                mcts.memory.add(state, policy, np.array([value]))
            
            if (game_idx + 1) % 25 == 0:
                print(f"  Games: {game_idx + 1}/{num_games_per_episode} | Buffer: {len(mcts.memory)}")

        # 2. TRAIN NETWORK
        if len(mcts.memory) > batch_size:
            print(f"  Training... ", end="", flush=True)
            mcts.train_network(num_batches=50)
            print("done")
        
        # 3. Save checkpoint
        if (ep + 1) % 5 == 0:
            torch.save(mcts.model.state_dict(), f"model_checkpoint_ep{ep+1}.pt")
            print(f"  Saved: model_checkpoint_ep{ep+1}.pt\n")

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    episodes = 50
    num_games_per_episode = 10
    batch_size = 64
    num_simulations = 200

    run_training(
        episodes=episodes,
        num_games_per_episode=num_games_per_episode,
        batch_size=batch_size,
        num_simulations=num_simulations
    )