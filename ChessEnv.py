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

class ChessEnv:
    def __init__(self):
        self.start_fen = "8/2ppk3/8/8/8/8/PPP1K3/8 w - - 0 1"
        self.board = chess.Board(self.start_fen)


    def reset(self):
        self.board.reset(self.start_fen)
        return self.get_state()


    # GET STATE (20 channel tensor)
    def get_state(self):
        state = np.zeros((20, 8, 8), dtype=np.float32)

        board = self.board
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
    

    def apply_action(self, action):
        """
        action: (8, 8, 73) policy output
        """

        board = self.board
        mover = board.turn
        flip = (mover == chess.BLACK)

        # 1. Select action by taking argmax of the logits
        idx = np.argmax(action)
        row, col, move_type = np.unravel_index(idx, (8, 8, 73))

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
            return True, mover
        else:
            return False, mover


    # LEGAL ACTIONS
    def get_legal_actions(self):
        return list(self.board.legal_moves)

    
    # LEGAL ACTION MASK
    def get_action_mask(self):
        """Returns a boolean (8, 8, 73) mask of legal moves."""
        mask = np.zeros((8, 8, 73), dtype=bool)
        flip = (self.board.turn == chess.BLACK)
        for move in self.board.legal_moves:
            idx = self.move_to_index(move, flip)
            if idx is not None:
                r, c, t = np.unravel_index(idx, (8, 8, 73))
                mask[r, c, t] = True
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