import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ChessEnv import ChessEnv
from Neuralnet import CNNNet
from utils import ReplayBuffer
import chess

NUM_SIMULATIONS = 100


class Node:
    def __init__(self, state):
        self.state = state
        self.P = None   # prior probabilities
        self.N = np.zeros(4672)  # how many times each action has been picked
        self.W = np.zeros(4672)  # cumulative value
        self.Q = np.zeros(4672)  # current Q-value for each action
        self.children = {}
        self.is_expanded = False


class MCTS:
    def __init__(self, env: ChessEnv = None, buffer_size=100000):
        self.env = env if env is not None else ChessEnv()
        self.c = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayBuffer(capacity=buffer_size)

    def mask_illegal_moves(self, actions, state):
        """Mask illegal moves by setting their values to 0."""
        action_mask = ChessEnv.get_action_mask(state)
        # Flatten mask from (8, 8, 73) to (4672,)
        action_mask = action_mask.reshape(-1)
        masked_actions = np.where(action_mask, actions, -np.inf)
        return masked_actions

    def select(self, node):
        """Selection phase: traverse tree using UCB until unexpanded node."""
        while node.is_expanded:
            state = node.state
            total_n = np.sum(node.N)

            ucb = node.Q + self.c * node.P * (np.sqrt(total_n + 1) / (1 + node.N))

            # Mask illegal moves
            ucb = self.mask_illegal_moves(ucb, state)

            action = np.argmax(ucb)
            if action not in node.children:
                return node, action

            node = node.children[action]

        return node, None

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

        p = self.mask_illegal_moves(p, state)
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

    def run_simulation(self, root):
        """Run a single MCTS simulation."""
        node = root
        path = []

        # Selection phase
        while node.is_expanded:
            total_N = np.sum(node.N)
            ucb = node.Q + self.c * node.P * np.sqrt(total_N + 1) / (1 + node.N)
            ucb = self.mask_illegal_moves(ucb, node.state)

            action = np.argmax(ucb)
            path.append((node, action))

            if action not in node.children:
                next_state = self.step_environment(node.state, action)
                node.children[action] = Node(next_state)
                node = node.children[action]
                break

            node = node.children[action]

        # Expansion & Evaluation
        value = self.expand(node)

        # Backpropagation
        self.backpropagate(path, value)

    def get_policy(self, root, temperature=1.0):
        """Get policy distribution from visit counts."""
        N = root.N.copy().astype(np.float32)

        if temperature == 0:
            pi = np.zeros_like(N)
            pi[np.argmax(N)] = 1
            return pi

        N = N ** (1 / temperature)
        return N / (np.sum(N) + 1e-10)

    def play_game(self):
        """Play a complete game of self-play."""
        memory = []
        board = self.env.reset()
        state = ChessEnv.encode_state(board)

        root = Node(state)
        move_count = 0
        max_moves = 500  # Prevent infinite games

        while not board.is_game_over() and move_count < max_moves:
            move_count += 1

            # Run MCTS
            for _ in range(NUM_SIMULATIONS):
                self.run_simulation(root)

            # Get policy
            pi = self.get_policy(root, temperature=1.0)

            # Store state and policy
            memory.append((state.copy(), pi.copy()))

            # Sample a move
            action = np.random.choice(len(pi), p=pi)

            # Step environment
            move = self.action_to_move(board, action)
            if move is None or move not in board.legal_moves:
                break

            board.push(move)
            state = ChessEnv.encode_state(board)

            # Reuse tree to speed up
            if action in root.children:
                root = root.children[action]
            else:
                root = Node(state)

        # Game ended so assign rewards
        z = self.get_game_result(board)  # +1 / -1 / 0

        # Assign reward to each step
        data = []
        for i, (s, pi) in enumerate(memory):
            # Alternate perspective
            if i % 2 == 0:
                value = z
            else:
                value = -z

            data.append((s, pi, value))

        return data

    def train_network(self, num_batches=10):
        """Train the neural network on data from replay buffer."""
        for _ in range(num_batches):
            batch = self.memory.sample(batch_size=32)

            states, target_pis, target_vs = batch

            # Convert to tensors
            states_tensor = torch.FloatTensor(states).to(self.device)
            target_pis_tensor = torch.FloatTensor(target_pis).to(self.device)
            target_vs_tensor = torch.FloatTensor(target_vs).to(self.device)

            # Forward pass
            pred_p, pred_v = self.model(states_tensor)

            # Flatten policy output from (batch, 8, 8, 73) to (batch, 4672)
            pred_p = pred_p.view(pred_p.size(0), -1)

            # Compute loss
            value_loss = nn.MSELoss()(pred_v, target_vs_tensor)

            # Policy loss (cross-entropy)
            log_probs = torch.log_softmax(pred_p, dim=1)
            policy_loss = -(target_pis_tensor * log_probs).sum(dim=1).mean()

            loss = value_loss + policy_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def step_environment(self, state, action):
        """Execute action in environment and return new state."""
        board = ChessEnv.decode_state(state)
        move = self.action_to_move(board, action)

        if move is None or move not in board.legal_moves:
            return state

        board.push(move)
        return ChessEnv.encode_state(board)

    def action_to_move(self, board, action):
        """Convert action index to chess move."""
        legal_moves = list(board.legal_moves)

        if action < len(legal_moves):
            return legal_moves[action]

        if legal_moves:
            return legal_moves[0]
        return None

    def get_game_result(self, board):
        """Get result of game: 1 for white win, -1 for black win, 0 for draw."""
        if not board.is_game_over():
            return 0

        result = board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0


if __name__ == "__main__":
    print("Initializing MCTS Chess Training Engine...")

    episodes = 10
    num_games_per_episode = 50

    # Initialize MCTS agent
    mcts = MCTS(buffer_size=100000)

    print("Starting MCTS Self-Play Training Loop...")

    for ep in range(episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}/{episodes}")
        print(f"{'='*50}")

        # 1. SELF-PLAY
        print(f"Playing {num_games_per_episode} games...")
        games_data = []

        for game_idx in range(num_games_per_episode):
            game_data = mcts.play_game()
            games_data.extend(game_data)
            if (game_idx + 1) % 10 == 0:
                print(f"  Completed {game_idx + 1}/{num_games_per_episode} games")

        # Add all game data to replay buffer
        for state, policy, value in games_data:
            mcts.memory.add(state, policy, np.array([value]))

        print(f"Replay buffer size: {len(mcts.memory)}")

        # 2. TRAIN NETWORK
        if len(mcts.memory) > 32:
            print("Training network...")
            mcts.train_network(num_batches=20)
            print("Training complete")
        else:
            print(f"Not enough data in replay buffer (need 32, have {len(mcts.memory)})")

        # 3. Save model checkpoint
        if (ep + 1) % 5 == 0:
            torch.save(mcts.model.state_dict(), f"model_checkpoint_ep{ep+1}.pt")
            print(f"Model saved: model_checkpoint_ep{ep+1}.pt")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)