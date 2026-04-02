import numpy as np
import chess
from ChessEnv import ChessEnv


class Node:
    """Simple MCTS Node without neural network."""
    def __init__(self, board=None, parent=None, parent_move=None):
        self.board = board.copy() if board else None
        self.parent = parent
        self.parent_move = parent_move  # The move that led to this node from parent
        self.children = {}  # Maps move to child node
        self.visits = 0
        self.value_sum = 0.0
        self.untried_moves = None
    
    def get_untried_moves(self):
        """Get legal moves that haven't been explored yet."""
        if self.untried_moves is None:
            self.untried_moves = list(self.board.legal_moves)
        return self.untried_moves
    
    def get_ucb_value(self, c=1.414):
        """Calculate UCB value for this node."""
        if self.parent is None:
            return float('inf')
        
        exploitation = self.value_sum / (self.visits + 1e-10)
        exploration = c * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1e-10))
        return exploitation + exploration
    
    def has_untried_moves(self):
        """Check if there are unexplored moves."""
        return len(self.get_untried_moves()) > 0


class MCTS_simple:
    """
    Simple Monte Carlo Tree Search without neural networks.
    Uses ChessEnv.get_potential() as the evaluation function.
    """
    
    def __init__(self, num_simulations, c=2.0):
        """
        Args:
            num_simulations: Number of MCTS simulations to run
            c: Exploration constant for UCB formula
        """
        self.num_simulations = num_simulations
        self.c = c
    
    def best_move(self, board):
        """
        Find the best move for a given board position using MCTS.
        
        Args:
            board: chess.Board object
            
        Returns:
            The best move as a chess.Move object
        """
        root = Node(board=board)
        
        # Run simulations
        for i in range(self.num_simulations):
            if i%1000 == 0:
                print("i:", i)
            self._simulate(root)
        
        # Return move with highest visit count
        if not root.children:
            # No children explored (shouldn't happen), return random legal move
            return list(board.legal_moves)[0] if board.legal_moves.count() > 0 else None
        
        best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
        return best_move
    
    def _simulate(self, node):
        """
        Run one complete simulation:
        1. Selection: traverse tree using UCB
        2. Expansion: expand one new node
        3. Rollout: random playout from the new node
        4. Backpropagation: update statistics
        
        Args:
            node: Root node to start simulation from
        """
        # Selection & Expansion
        current_node = node
        
        # Selection phase: traverse until we hit a node with untried moves or leaf
        while not current_node.board.is_game_over() and not current_node.has_untried_moves():
            # All moves tried, select best child by UCB
            if current_node.children:
                move, child_node = max(
                    current_node.children.items(),
                    key=lambda x: x[1].get_ucb_value(self.c)
                )
                current_node = child_node
            else:
                break
        
        # Expansion phase: try one untried move
        if not current_node.board.is_game_over() and current_node.has_untried_moves():
            untried_moves = current_node.get_untried_moves()
            # Use pop() to remove the move from the list of untried moves
            move = untried_moves.pop(np.random.randint(len(untried_moves)))
            
            new_board = current_node.board.copy()
            new_board.push(move)
            child_node = Node(board=new_board, parent=current_node, parent_move=move)
            current_node.children[move] = child_node
            current_node = child_node
        
        # Rollout phase: random playout to terminal or use potential
        value = self._rollout(current_node)
        
        # Backpropagation: update statistics along the path
        self._backpropagate(current_node, value)
    
    def _rollout(self, node):
        """
        Args:
            node: Node to evaluate
            
        Returns:
            Value estimate in range [-1, 1]
        """
        return ChessEnv.get_potential(node.board)
    
    def _backpropagate(self, node, value):
        """
        Update statistics along the path from node to root.
        Alternates the perspective of the value.
        
        Args:
            node: Node to start backpropagation from
            value: Value to backpropagate (from current player's perspective)
        """
        if node.board.turn == chess.BLACK: 
            v = value      # High is good for White
        else:
            v = -value     # High is good for Black
        
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value_sum += v
            
            # Flip for the parent: if it's good for me, it's bad for the person 
            # who moved to get me here.
            v = -v 
            current_node = current_node.parent
    
    def get_move_statistics(self, board):
        """
        Get statistics for all moves (visit counts and average values).
        Useful for analysis.
        
        Args:
            board: chess.Board object
            
        Returns:
            Dictionary mapping moves to (visits, avg_value) tuples
        """
        root = Node(board=board)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Collect statistics
        stats = {}
        for move, child_node in root.children.items():
            avg_value = child_node.value_sum / (child_node.visits + 1e-10)
            stats[move.uci()] = {
                'visits': child_node.visits,
                'avg_value': avg_value,
                'move': move
            }
        
        return stats
