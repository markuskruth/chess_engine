import chess
import numpy as np

DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1)
]

KNIGHT_MOVES = [
    (1, 2), (2, 1), (-1, 2), (-2, 1),
    (1, -2), (2, -1), (-1, -2), (-2, -1)
]

PROMOTIONS = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

START_FEN = "4k3/7R/K7/R7/8/8/8/8 w - - 0 1"

class ChessEnv:
    def __init__(self):
        self.start_fen = "8/2ppk3/8/8/8/8/PPP1K3/8 w - - 0 1"
        self.board = chess.Board(START_FEN)


    def reset(self):
        self.board.set_fen(START_FEN)
        return self.board


    # Decode state to board
    @staticmethod
    def decode_state(state):
        # Initialize an empty board (no pieces)
        board = chess.Board(None)
        
        # 1. Decode Pieces (Channels 0-11)
        # 0-5: White P, N, B, R, Q, K | 6-11: Black P, N, B, R, Q, K
        for channel in range(12):
            piece_type = (channel % 6) + 1
            color = chess.WHITE if channel < 6 else chess.BLACK
            
            # Find indices where piece exists (value == 1)
            rows, cols = np.where(state[channel] == 1)
            
            for r, c in zip(rows, cols):
                # Reverse: tensor_row = 7 - row  =>  row = 7 - tensor_row
                row = 7 - r
                col = c
                square = chess.square(col, row)
                board.set_piece_at(square, chess.Piece(piece_type, color))

        # 2. Set Turn
        # Since the agent always sees themselves as White, the decoded board 
        # turn is always White.
        board.turn = chess.WHITE

        # 3. Castling Rights (Channels 13-16)
        # We use bitwise OR to set rights using chess library bitboards
        rights = 0
        if np.any(state[13] == 1): rights |= chess.BB_H1 # White King-side
        if np.any(state[14] == 1): rights |= chess.BB_A1 # White Queen-side
        if np.any(state[15] == 1): rights |= chess.BB_H8 # Black King-side
        if np.any(state[16] == 1): rights |= chess.BB_A8 # Black Queen-side
        board.castling_rights = rights

        # 4. En Passant Square (Channel 17)
        ep_rows, ep_cols = np.where(state[17] == 1)
        if len(ep_rows) > 0:
            row = 7 - ep_rows[0]
            col = ep_cols[0]
            board.ep_square = chess.square(col, row)

        # 5. Halfmove Clock (Channel 18)
        # Multiply back by 50 and round to nearest int
        board.halfmove_clock = int(np.round(state[18, 0, 0] * 50))

        # Note: 3-fold repetition (Channel 19) cannot be fully "restored" 
        # into a new Board object's move stack easily, but the board 
        # state itself is now valid for move generation.

        return board

    # Encode state from board (20 channel tensor)
    @staticmethod
    def encode_state(board):
        state = np.zeros((20, 8, 8), dtype=np.float32)
        flip = (board.turn == chess.BLACK)

        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            row = chess.square_rank(square)
            col = chess.square_file(square)

            # Flip board if black to move
            if flip:
                row = 7 - row

            # Convert to tensor coords
            tensor_row = 7 - row
            tensor_col = col

            # Swap colors if flipped
            if flip:
                piece_color = not piece.color
            else:
                piece_color = piece.color

            piece_type = piece.piece_type - 1
            color_offset = 0 if piece_color == chess.WHITE else 6
            channel = piece_type + color_offset

            state[channel, tensor_row, tensor_col] = 1

        # Turn (always "white")
        # The agent always sees themselves as white
        state[12, :, :] = 1  # always 1

        # Castling rights
        if not flip:
            state[13, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
            state[14, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
            state[15, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
            state[16, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))
        else:
            # swap roles
            state[13, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
            state[14, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))
            state[15, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
            state[16, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))

        # En passant
        if board.ep_square is not None:
            row = chess.square_rank(board.ep_square)
            col = chess.square_file(board.ep_square)

            if flip:
                row = 7 - row

            tensor_row = 7 - row
            tensor_col = col

            state[17, tensor_row, tensor_col] = 1

        # Halfmove clock (50 move rule)
        state[18, :, :] = board.halfmove_clock / 50.0

        # 3-fold repetition
        rep_count = 0
        if board.is_repetition(2): rep_count = 1
        if board.is_repetition(3): rep_count = 2
        state[19, :, :] = rep_count / 2.0

        return state
    
    @staticmethod
    def apply_action(action_idx, board):
        """
        action: (8, 8, 73) policy output
        """

        mover = board.turn
        flip = (mover == chess.BLACK)

        # 1. Select action by taking argmax of the logits
        row, col, move_type = np.unravel_index(action_idx, (8, 8, 73))

        # Convert FROM square to real board coords
        real_row = 7 - row if flip else row
        real_col = col

        from_square = chess.square(real_col, 7 - real_row)

        # 2. Decode move
        # Sliding moves (0–55)
        if move_type < 56:
            direction_idx = move_type // 7
            distance = (move_type % 7) + 1

            dx, dy = DIRECTIONS[direction_idx]

            # Flip vertical direction if needed
            if flip:
                dy = -dy

            to_col = col + dx * distance
            to_row = row - dy * distance  # still in agent coords

            # Convert to real board coords
            real_to_row = 7 - to_row if flip else to_row
            real_to_col = to_col

            if 0 <= to_col < 8 and 0 <= to_row < 8:
                to_square = chess.square(real_to_col, 7 - real_to_row)
                move = chess.Move(from_square, to_square)
            else:
                return False, mover

        # Knight moves (56–63)
        elif move_type < 64:
            k_idx = move_type - 56
            dx, dy = KNIGHT_MOVES[k_idx]

            if flip:
                dy = -dy

            to_col = col + dx
            to_row = row - dy

            real_to_row = 7 - to_row if flip else to_row
            real_to_col = to_col

            if 0 <= to_col < 8 and 0 <= to_row < 8:
                to_square = chess.square(real_to_col, 7 - real_to_row)
                move = chess.Move(from_square, to_square)
            else:
                return False, mover

        # Promotions (64–72)
        else:
            promo_idx = move_type - 64

            direction = promo_idx // 3
            promo_piece = PROMOTIONS[promo_idx % 3]

            # In agent space: always moving "up"
            dy = -1
            if flip:
                dy = 1

            if direction == 0:
                dx = 0
            elif direction == 1:
                dx = -1
            else:
                dx = 1

            to_col = col + dx
            to_row = row + dy

            real_to_row = 7 - to_row if flip else to_row
            real_to_col = to_col

            if 0 <= to_col < 8 and 0 <= to_row < 8:
                to_square = chess.square(real_to_col, 7 - real_to_row)
                move = chess.Move(from_square, to_square, promotion=promo_piece)
            else:
                return False, mover


        # 3. Validate move
        if move in board.legal_moves:
            board.push(move)
            return True, move
        else:
            return False, mover


    # LEGAL ACTIONS
    def get_legal_actions(self):
        return list(self.board.legal_moves)

    
    # LEGAL ACTION MASK
    @staticmethod
    def get_action_mask(state=None, board=None):
        """Returns a boolean (8, 8, 73) mask of legal moves."""
        if state is not None:
            board = ChessEnv.decode_state(state)

        mask = np.zeros((8, 8, 73), dtype=bool)

        for move in board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            
            from_row, from_col = chess.square_rank(from_sq), chess.square_file(from_sq)
            to_row, to_col = chess.square_rank(to_sq), chess.square_file(to_sq)
            
            # Convert to your tensor coordinates (tensor_row = 7 - rank)
            t_row, t_col = 7 - from_row, from_col
            
            dr, dc = to_row - from_row, to_col - from_col
            plane = -1

            # 1. Under-promotions (Planes 64-72)
            # Queen promotions are handled as normal moves in planes 0-63
            if move.promotion and move.promotion != chess.QUEEN:
                # 3 directions: capture-left (-1), forward (0), capture-right (1)
                promo_dir = dc + 1 
                # 3 pieces: Knight (0), Bishop (1), Rook (2)
                promo_piece = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
                plane = 64 + (promo_dir * 3) + promo_piece

            # 2. Knight Moves (Planes 56-63)
            elif (dr, dc) in KNIGHT_MOVES:
                plane = 56 + KNIGHT_MOVES.index((dr, dc))

            # 3. Queen-like Moves (Planes 0-55)
            else:
                distance = max(abs(dr), abs(dc))
                direction = (dr // distance, dc // distance)
                if direction in DIRECTIONS:
                    dir_idx = DIRECTIONS.index(direction)
                    plane = (dir_idx * 7) + (distance - 1)

            if plane != -1:
                mask[t_row, t_col, plane] = True

        return mask

    # STEP FUNCTION
    def step(self, action):
        """
        action: (8x8x73)
        """

        valid_action = self.apply_action(action)
        if not valid_action:
            raise ValueError("Illegal move")

        done = self.board.is_game_over()
        reward = self.get_reward()

        return self.get_state(), reward, done, {}


    # REWARD FUNCTION
    def get_reward(self):
        if not self.board.is_game_over():
            return 0

        result = self.board.result()

        just_moved = not self.board.turn
        if result == "1-0":
            return 1 if just_moved == chess.WHITE else -1
        elif result == "0-1":
            return 1 if just_moved == chess.BLACK else -1
        else:
            return 0

    def render(self):
        print(self.board)
    
    def get_state(self):
        """Return encoded state from the current board."""
        return ChessEnv.encode_state(self.board)