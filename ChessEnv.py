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

#START_FEN = "4k3/7R/K7/R7/8/8/8/8 w - - 0 1"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# ---------------------------------------------------------------------------
# Piece-Square Tables (PSTs) — from White's perspective, index = chess square
# (a1=0 .. h1=7, a2=8 .. h8=63). Use chess.square_mirror(sq) for Black.
# Two tables per piece type: middlegame (MG) and endgame (EG).
# Source: PeSTO / CPW Simplified Evaluation Function, standard tuned values.
# ---------------------------------------------------------------------------

_PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_KNIGHT_MG = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]
_KNIGHT_EG = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
_BISHOP_MG = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
_BISHOP_EG = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]
_ROOK_MG = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]
_ROOK_EG = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8,  -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]
_QUEEN_MG = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
_QUEEN_EG = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]
_KING_MG = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
_KING_EG = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

# Map piece type to (mg_table, eg_table)
_PST = {
    chess.PAWN:   (_PAWN_MG,   _PAWN_EG),
    chess.KNIGHT: (_KNIGHT_MG, _KNIGHT_EG),
    chess.BISHOP: (_BISHOP_MG, _BISHOP_EG),
    chess.ROOK:   (_ROOK_MG,   _ROOK_EG),
    chess.QUEEN:  (_QUEEN_MG,  _QUEEN_EG),
    chess.KING:   (_KING_MG,   _KING_EG),
}

# Material values (mg, eg) in centipawns
_MAT_MG = {chess.PAWN: 82,  chess.KNIGHT: 337, chess.BISHOP: 365,
           chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0}
_MAT_EG = {chess.PAWN: 94,  chess.KNIGHT: 281, chess.BISHOP: 297,
           chess.ROOK: 512, chess.QUEEN: 936,  chess.KING: 0}

# Phase weights for tapering (King excluded — doesn't count toward phase)
_PHASE_WEIGHTS = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
                  chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
_TOTAL_PHASE = 16 * 0 + 4 * 1 + 4 * 1 + 4 * 2 + 2 * 4  # = 24


class ChessEnv:

    @staticmethod
    def reset():
        return chess.Board(START_FEN)


    @staticmethod
    def get_evaluation(board):
        """
        Tapered evaluation function from White's perspective, normalized to [-1, 1].

        Components:
          - Tapered material + piece-square tables (PeSTO values, mg/eg interpolated)
          - Bishop pair bonus
          - Pawn structure: passed pawns, isolated pawns, doubled pawns
          - Rook bonuses: open file, half-open file, 7th rank
          - King safety: pawn shelter + open files near king
          - Symmetric mobility (both sides)
        """
        if board.is_game_over():
            return ChessEnv.get_reward(board)

        mg_score = 0  # middlegame accumulator (White positive)
        eg_score = 0  # endgame accumulator
        phase    = 0  # for tapering: 0=endgame, _TOTAL_PHASE=opening

        # ── 1. MATERIAL + PST ───────────────────────────────────────────────
        for sq, piece in board.piece_map().items():
            pt  = piece.piece_type
            mg_t, eg_t = _PST[pt]
            # White: use square directly; Black: mirror rank
            idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
            sign = 1 if piece.color == chess.WHITE else -1
            mg_score += sign * (_MAT_MG[pt] + mg_t[idx])
            eg_score += sign * (_MAT_EG[pt] + eg_t[idx])
            phase    += _PHASE_WEIGHTS[pt]

        # Taper: phase=24 → pure MG, phase=0 → pure EG
        phase     = min(phase, _TOTAL_PHASE)
        cp_score  = (mg_score * phase + eg_score * (_TOTAL_PHASE - phase)) // _TOTAL_PHASE

        # ── 2. BISHOP PAIR ──────────────────────────────────────────────────
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            cp_score += 30
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            cp_score -= 30

        # ── 3. PAWN STRUCTURE ───────────────────────────────────────────────
        for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
            pawns     = board.pieces(chess.PAWN, color)
            opp_pawns = board.pieces(chess.PAWN, not color)

            pawn_files = [chess.square_file(sq) for sq in pawns]

            for sq in pawns:
                f = chess.square_file(sq)
                r = chess.square_rank(sq)

                # Doubled pawns: another friendly pawn on the same file
                if pawn_files.count(f) > 1:
                    cp_score -= sign * 20

                # Isolated pawns: no friendly pawn on adjacent files
                adj_files = chess.BB_FILES[f - 1] if f > 0 else 0
                adj_files |= chess.BB_FILES[f + 1] if f < 7 else 0
                if not (pawns & adj_files):
                    cp_score -= sign * 20

                # Passed pawn: no enemy pawn on same or adjacent files ahead
                if color == chess.WHITE:
                    ahead_mask = chess.BB_FILES[f] & ~chess.BB_RANKS[r]
                    for rf in range(r + 1, 8):
                        ahead_mask |= chess.BB_SQUARES[chess.square(f, rf)]
                    if f > 0:
                        for rf in range(r + 1, 8):
                            ahead_mask |= chess.BB_SQUARES[chess.square(f - 1, rf)]
                    if f < 7:
                        for rf in range(r + 1, 8):
                            ahead_mask |= chess.BB_SQUARES[chess.square(f + 1, rf)]
                    is_passed = not bool(opp_pawns & ahead_mask)
                    rank_bonus = [0, 10, 20, 40, 60, 90, 120, 0]
                    advance_rank = r
                else:
                    ahead_mask = 0
                    for rf in range(0, r):
                        ahead_mask |= chess.BB_SQUARES[chess.square(f, rf)]
                        if f > 0:
                            ahead_mask |= chess.BB_SQUARES[chess.square(f - 1, rf)]
                        if f < 7:
                            ahead_mask |= chess.BB_SQUARES[chess.square(f + 1, rf)]
                    is_passed = not bool(opp_pawns & ahead_mask)
                    rank_bonus = [0, 120, 90, 60, 40, 20, 10, 0]
                    advance_rank = 7 - r

                if is_passed:
                    cp_score += sign * rank_bonus[advance_rank]

        # ── 4. ROOK BONUSES ─────────────────────────────────────────────────
        for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
            seventh_rank = 6 if color == chess.WHITE else 1
            for sq in board.pieces(chess.ROOK, color):
                f = chess.square_file(sq)
                r = chess.square_rank(sq)
                file_bb = chess.BB_FILES[f]
                own_pawns = board.pieces(chess.PAWN, color)
                opp_pawns = board.pieces(chess.PAWN, not color)
                # Open file: no pawns of either color
                if not (own_pawns | opp_pawns) & file_bb:
                    cp_score += sign * 20
                # Half-open file: no friendly pawn, but an enemy pawn
                elif not own_pawns & file_bb:
                    cp_score += sign * 10
                # Rook on 7th rank
                if r == seventh_rank:
                    cp_score += sign * 30

        # ── 5. KING SAFETY ──────────────────────────────────────────────────
        for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
            king_sq   = board.king(color)
            if king_sq is None:
                continue
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            own_pawns = board.pieces(chess.PAWN, color)

            # Open / half-open files near king
            for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                file_bb = chess.BB_FILES[f]
                if not own_pawns & file_bb:
                    # Fully open file (no own pawn) — bigger penalty
                    opp_pawns = board.pieces(chess.PAWN, not color)
                    cp_score -= sign * (50 if not opp_pawns & file_bb else 25)

            # Pawn shelter: three squares directly in front of the king
            shelter_rank = king_rank + (1 if color == chess.WHITE else -1)
            if 0 <= shelter_rank <= 7:
                for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                    shelter_sq = chess.square(f, shelter_rank)
                    if not own_pawns & chess.BB_SQUARES[shelter_sq]:
                        cp_score -= sign * 20

        # ── 6. SYMMETRIC MOBILITY ───────────────────────────────────────────
        # Measure mobility for both sides without modifying the board
        white_mob = sum(1 for _ in board.legal_moves) if board.turn == chess.WHITE \
                    else sum(1 for _ in board.generate_legal_moves())
        # Temporarily flip turn to count opponent moves
        board.turn = not board.turn
        black_mob = sum(1 for _ in board.legal_moves)
        board.turn = not board.turn  # restore

        cp_score += 5 * (white_mob - black_mob)

        # ── 7. NORMALIZE to [-1, 1] ─────────────────────────────────────────
        # formula: 2 / (1 + 10^(-cp / 400)) - 1
        win_prob = 2 / (1 + 10 ** (-cp_score / 400)) - 1
        return win_prob


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

            to_col = col + dx * distance
            to_row = row - dy * distance  # still in agent coords

            # Convert to real board coords
            real_to_row = 7 - to_row if flip else to_row
            real_to_col = to_col

            if 0 <= to_col < 8 and 0 <= to_row < 8:
                to_square = chess.square(real_to_col, 7 - real_to_row)
                # Queen promotion: pawn reaching the back rank via a queen-like move
                promo = None
                piece = board.piece_at(from_square)
                if piece and piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(to_square)
                    if to_rank == 7 or to_rank == 0:
                        promo = chess.QUEEN
                move = chess.Move(from_square, to_square, promotion=promo)
            else:
                return False, mover

        # Knight moves (56–63)
        elif move_type < 64:
            k_idx = move_type - 56
            dx, dy = KNIGHT_MOVES[k_idx]

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

            # In agent space: always moving "up" (decreasing tensor_row)
            dy = -1

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
    @staticmethod
    def get_legal_actions(self):
        return list(self.board.legal_moves)

    
    # LEGAL ACTION MASK
    @staticmethod
    def get_action_mask(state=None, board=None):
        """Returns a boolean (8, 8, 73) mask of legal moves.

        The mask is always in the current player's coordinate frame (same
        frame as encode_state and apply_action):
          - tensor_row = real_rank  for BLACK  (board is flipped vertically)
          - tensor_row = 7-real_rank for WHITE
        The rank-difference direction is also flipped for black so that
        DIRECTIONS and KNIGHT_MOVES can be looked up uniformly.
        """
        if state is not None:
            board = ChessEnv.decode_state(state)

        flip = (board.turn == chess.BLACK)
        mask = np.zeros((8, 8, 73), dtype=bool)

        for move in board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square

            from_rank = chess.square_rank(from_sq)
            from_file = chess.square_file(from_sq)
            to_rank   = chess.square_rank(to_sq)
            to_file   = chess.square_file(to_sq)

            dc = to_file - from_file  # file diff is the same in both views

            if flip:
                # For black: tensor_row = real_rank (double flip cancels)
                t_row = from_rank
                # Rank direction is inverted in the agent's view
                dr_agent = -(to_rank - from_rank)
            else:
                # For white: tensor_row = 7 - real_rank
                t_row = 7 - from_rank
                dr_agent = to_rank - from_rank

            t_col = from_file
            plane = -1

            # 1. Under-promotions (Planes 64-72)
            # Queen promotions fall through to the queen-move planes (0-55)
            if move.promotion and move.promotion != chess.QUEEN:
                # direction: 0=forward(dc=0), 1=capture-left(dc=-1), 2=capture-right(dc=1)
                if dc == 0:
                    promo_dir = 0
                elif dc == -1:
                    promo_dir = 1
                else:
                    promo_dir = 2
                # Piece encoding matches PROMOTIONS = [ROOK, BISHOP, KNIGHT]
                promo_piece = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[move.promotion]
                plane = 64 + (promo_dir * 3) + promo_piece

            # 2. Knight Moves (Planes 56-63)
            # KNIGHT_MOVES stores (dx, dy) = (file_diff, agent_rank_diff)
            elif (dc, dr_agent) in KNIGHT_MOVES:
                plane = 56 + KNIGHT_MOVES.index((dc, dr_agent))

            # 3. Queen-like Moves (Planes 0-55)
            # DIRECTIONS stores (dx, dy) = (file_diff, agent_rank_diff)
            else:
                distance = max(abs(dr_agent), abs(dc))
                direction = (dc // distance, dr_agent // distance)
                if direction in DIRECTIONS:
                    dir_idx = DIRECTIONS.index(direction)
                    plane = (dir_idx * 7) + (distance - 1)

            if plane != -1:
                mask[t_row, t_col, plane] = True

        return mask

    # STEP FUNCTION
    @staticmethod
    def step(board, action_idx):
        """
        action: (8x8x73)
        """

        valid_action = ChessEnv.apply_action(action_idx, board)
        if not valid_action:
            raise ValueError("Illegal move")

        done = board.is_game_over()
        reward = ChessEnv.get_reward()

        return ChessEnv.get_state(), reward, done, {}


    # REWARD FUNCTION
    @staticmethod
    def get_reward(board):
        if not board.is_game_over():
            return 0

        result = board.result()

        just_moved = not board.turn
        if result == "1-0":
            return 10 if just_moved == chess.WHITE else -10
        elif result == "0-1":
            return 10 if just_moved == chess.BLACK else -10
        else:
            return 0

    def render(self):
        print(self.board)
    
    def get_state(self):
        """Return encoded state from the current board."""
        return ChessEnv.encode_state(self.board)