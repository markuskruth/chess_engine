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

        # Reward params
        self.gamma = 0.99  # Discount factor
        self.lambd = 0.5   # Shaping weight (anneal this to 0 over time)

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
            ucb = self.mask_illegal_moves(ucb, board=board)

            action = np.argmax(ucb)
            path.append((node, action))

            if action not in node.children:
                # Create new child node
                legal_moves = list(board.legal_moves)
                if action < len(legal_moves):
                    move = legal_moves[action]
                    board.push(move)
                    state = ChessEnv.encode_state(board)
                    new_node = Node(state, board.copy())
                    new_node.legal_moves_cache = self.get_legal_moves_array(board)
                    node.children[action] = new_node
                    node = new_node
                break
            else:
                # Traverse existing child
                legal_moves = list(board.legal_moves)
                if action < len(legal_moves):
                    board.push(legal_moves[action])
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

        p = self.mask_illegal_moves(p, state=state)
        # Convert -inf back to 0 and normalize
        p = np.where(np.isinf(p), 0, p)
        p = np.exp(p) if np.max(p) > 0 else p  # Handle log probabilities
        p = p / (np.sum(p) + 1e-10)  # normalize with epsilon

        node.P = p
        node.is_expanded = True

        return v


    def backpropagate(self, path, value):
        """Backpropagation phase: update statistics along the path."""
        for node, action in reversed(path):
            node.N[action] += 1
            node.W[action] += value
            node.Q[action] = node.W[action] / node.N[action]

            value = -value  # switch perspective


    def play_game(self, temperature=1.0):
        memory = []
        board = ChessEnv.reset()
        state = ChessEnv.encode_state(board)
        
        root = Node(state, board.copy())
        
        move_count = 0
        max_moves = 200
        trajectory = [] # Store (state, potential)

        while not board.is_game_over() and move_count < max_moves:
            phi_s = ChessEnv.get_potential(board) # Current state potential
            state = ChessEnv.encode_state(board)
            
            # Run MCTS simulations
            for _ in range(self.num_simulations):
                self.run_simulation(root)

            # Get policy and choose move
            N = root.N.copy().astype(np.float32)
            pi = N / (np.sum(N) + 1e-10)
            action_idx = np.random.choice(len(pi), p=pi)

            # Record state before move
            trajectory.append({
                "state": state,
                "pi": pi,
                "phi": phi_s,
                "turn": board.turn
            })

            # Execute move
            ChessEnv.apply_action(action_idx, board)
            move_count += 1

        # Calculate final outcome reward Z
        z = ChessEnv.get_reward(board)
        #z = self.get_game_result(board)
        
        # BACKFILL SHAPED REWARDS
        # The target value for training is the cumulative shaped return
        data = []
        for i in range(len(trajectory)):
            # Terminal reward is only added to the very last move
            r_terminal = z if (i == len(trajectory) - 1) else 0
            
            # PBRS Formula: F = gamma * Phi(s_next) - Phi(s_current)
            phi_curr = trajectory[i]["phi"]
            phi_next = trajectory[i+1]["phi"] if i+1 < len(trajectory) else 0
            
            shaping_term = (self.gamma * phi_next) - phi_curr
            shaped_reward = r_terminal + (self.lambd * shaping_term)
            
            # The network predicts the value from the current player's perspective
            # Flip sign if it's the opponent's turn relative to the game ender
            data.append((trajectory[i]["state"], trajectory[i]["pi"], shaped_reward))

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