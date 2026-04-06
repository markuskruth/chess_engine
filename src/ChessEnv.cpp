// ChessEnv.cpp — Phase 2 implementation of ChessEnv.h
//
// Produces byte-identical output to Python's ChessEnv.py for:
//   encode_state()      — (20, 8, 8) float32 tensor
//   get_action_mask()   — bool[4672] legal-move mask
//   apply_action()      — apply action index to board, return {valid, move}

#include "ChessEnv.h"
#include <stdexcept>
#include <cmath>
#include <string>
#include <algorithm>

// ── Move-encoding tables ──────────────────────────────────────────────────────
// Must match Python's DIRECTIONS, KNIGHT_MOVES, PROMOTIONS exactly.
//
// (dx, dy): dx = file-delta, dy = agent-frame rank-delta.
//   dy > 0 = "north" (toward the 8th rank in the agent's view).
//   Agent frame: tensor_row = 7 - real_rank; dy = real_rank_delta for white,
//                dy = -real_rank_delta for black.

// Python: DIRECTIONS = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1)]
const std::array<std::pair<int,int>, 8> DIRECTIONS = {{
    { 0,  1},   // 0: north
    { 0, -1},   // 1: south
    { 1,  0},   // 2: east
    {-1,  0},   // 3: west
    { 1,  1},   // 4: north-east
    {-1,  1},   // 5: north-west
    { 1, -1},   // 6: south-east
    {-1, -1},   // 7: south-west
}};

// Python: KNIGHT_MOVES = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
const std::array<std::pair<int,int>, 8> KNIGHT_MOVES = {{
    { 1,  2},   // 0
    { 2,  1},   // 1
    {-1,  2},   // 2
    {-2,  1},   // 3
    { 1, -2},   // 4
    { 2, -1},   // 5
    {-1, -2},   // 6
    {-2, -1},   // 7
}};

// Python: PROMOTIONS = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
const std::array<chess::PieceType, 3> UNDERPROMO_PIECES = {{
    chess::PieceType::ROOK,
    chess::PieceType::BISHOP,
    chess::PieceType::KNIGHT,
}};

// ── game_over ─────────────────────────────────────────────────────────────────

std::pair<bool, bool> ChessEnv::game_over(const chess::Board& board) {
    auto [reason, result] = board.isGameOver();
    bool over      = (result != chess::GameResult::NONE);
    bool checkmate = (reason == chess::GameResultReason::CHECKMATE);
    return {over, checkmate};
}

// ── encode_state ──────────────────────────────────────────────────────────────
//
// Agent always sees itself as White.
// When Black is to move the board is flipped vertically and colors swapped.
//
// Channels:
//   0-5:  current-player pieces  P N B R Q K  (PieceType value = channel index)
//   6-11: opponent pieces        P N B R Q K
//   12:   turn plane (always 1.0)
//   13:   current-player kingside  castling right
//   14:   current-player queenside castling right
//   15:   opponent kingside  castling right
//   16:   opponent queenside castling right
//   17:   en-passant square
//   18:   half-move clock / 50.0
//   19:   repetition count / 2.0  (0, 0.5, 1.0)

torch::Tensor ChessEnv::encode_state(const chess::Board& board) {
    auto state = torch::zeros({STATE_CHANNELS, BOARD_SZ, BOARD_SZ}, torch::kFloat32);
    float* data = state.data_ptr<float>();

    const bool flip = (board.sideToMove() == chess::Color::BLACK);

    // ── Channels 0-11: pieces ─────────────────────────────────────────────────
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece  piece = board.at(sq);
        if (piece == chess::Piece::NONE) continue;

        int rank_i = static_cast<int>(sq.rank());   // 0-7 (0 = rank 1)
        int file_i = static_cast<int>(sq.file());   // 0-7 (0 = file a)

        if (flip) rank_i = 7 - rank_i;
        int tensor_row = 7 - rank_i;   // tensor row 0 = back rank in agent view
        int tensor_col = file_i;

        // Swap colors when board is flipped
        chess::Color piece_color = flip ? ~piece.color() : piece.color();

        // chess-library PieceType: PAWN=0 .. KING=5  (= Python's piece_type - 1)
        int piece_type_idx = static_cast<int>(piece.type());
        int color_offset   = (piece_color == chess::Color::WHITE) ? 0 : 6;
        int channel        = piece_type_idx + color_offset;

        data[channel * 64 + tensor_row * 8 + tensor_col] = 1.0f;
    }

    // ── Channel 12: turn (always 1) ───────────────────────────────────────────
    for (int i = 0; i < 64; ++i) data[12 * 64 + i] = 1.0f;

    // ── Channels 13-16: castling rights ───────────────────────────────────────
    auto cr = board.castlingRights();
    using Side = chess::Board::CastlingRights::Side;

    chess::Color cur = flip ? chess::Color::BLACK : chess::Color::WHITE;
    chess::Color opp = flip ? chess::Color::WHITE : chess::Color::BLACK;

    float cur_ks = cr.has(cur, Side::KING_SIDE)  ? 1.0f : 0.0f;
    float cur_qs = cr.has(cur, Side::QUEEN_SIDE) ? 1.0f : 0.0f;
    float opp_ks = cr.has(opp, Side::KING_SIDE)  ? 1.0f : 0.0f;
    float opp_qs = cr.has(opp, Side::QUEEN_SIDE) ? 1.0f : 0.0f;

    for (int i = 0; i < 64; ++i) {
        data[13 * 64 + i] = cur_ks;
        data[14 * 64 + i] = cur_qs;
        data[15 * 64 + i] = opp_ks;
        data[16 * 64 + i] = opp_qs;
    }

    // ── Channel 17: en-passant square ─────────────────────────────────────────
    chess::Square ep_sq = board.enpassantSq();
    if (ep_sq != chess::Square::NO_SQ) {
        int ep_rank = static_cast<int>(ep_sq.rank());
        int ep_file = static_cast<int>(ep_sq.file());
        if (flip) ep_rank = 7 - ep_rank;
        int tensor_row = 7 - ep_rank;
        data[17 * 64 + tensor_row * 8 + ep_file] = 1.0f;
    }

    // ── Channel 18: half-move clock / 50.0 ───────────────────────────────────
    float hmc = static_cast<float>(board.halfMoveClock()) / 50.0f;
    for (int i = 0; i < 64; ++i) data[18 * 64 + i] = hmc;

    // ── Channel 19: repetition count / 2.0 ───────────────────────────────────
    // Python: rep_count = 0 / 1 / 2 if no-rep / 2-fold / 3-fold
    float rep = 0.0f;
    if (board.isRepetition(2)) rep = 0.5f;
    if (board.isRepetition(3)) rep = 1.0f;
    for (int i = 0; i < 64; ++i) data[19 * 64 + i] = rep;

    return state;
}

// ── get_action_mask ───────────────────────────────────────────────────────────
//
// Returns a flat bool[4672] mask (row-major: [t_row][t_col][plane]).
// Always in the current player's agent coordinate frame:
//   white: t_row = 7 - real_rank,   dr_agent =  (to_rank - from_rank)
//   black: t_row = real_rank,        dr_agent = -(to_rank - from_rank)
//
// Plane encoding (matches Python):
//   0-55:  queen-like slides — 8 directions × 7 distances
//   56-63: knight hops
//   64-72: under-promotions (3 directions × 3 piece types)
//   Queen promotions are encoded as queen-like 1-step forward moves.

ActionMask ChessEnv::get_action_mask(const chess::Board& board) {
    ActionMask mask{};  // zero-initialised

    const bool flip = (board.sideToMove() == chess::Color::BLACK);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    for (int mi = 0; mi < static_cast<int>(moves.size()); ++mi) {
        const chess::Move move = moves[mi];

        const int from_rank = static_cast<int>(move.from().rank());
        const int from_file = static_cast<int>(move.from().file());
        int to_rank         = static_cast<int>(move.to().rank());
        int to_file         = static_cast<int>(move.to().file());

        // Castling: chess-library stores move.to() as the rook square (e.g. e1→h1),
        // but the action space encodes the king's 2-square destination (e1→g1/c1).
        if (move.typeOf() == chess::Move::CASTLING) {
            to_file = (to_file > from_file) ? from_file + 2 : from_file - 2;
            to_rank = from_rank;  // king stays on same rank
        }

        const int dc = to_file - from_file;

        int t_row, dr_agent;
        if (flip) {
            t_row    = from_rank;
            dr_agent = -(to_rank - from_rank);
        } else {
            t_row    = 7 - from_rank;
            dr_agent = to_rank - from_rank;
        }
        const int t_col = from_file;

        int plane = -1;

        // 1. Under-promotions (planes 64-72)
        //    Queen promotions fall through to queen-like planes below.
        if (move.typeOf() == chess::Move::PROMOTION &&
            move.promotionType() != chess::PieceType::QUEEN) {

            int promo_dir;
            if      (dc ==  0) promo_dir = 0;
            else if (dc == -1) promo_dir = 1;
            else               promo_dir = 2;  // dc == 1

            int promo_piece_idx;
            chess::PieceType pt = move.promotionType();
            if      (pt == chess::PieceType::ROOK)   promo_piece_idx = 0;
            else if (pt == chess::PieceType::BISHOP)  promo_piece_idx = 1;
            else                                      promo_piece_idx = 2;  // KNIGHT

            plane = 64 + promo_dir * 3 + promo_piece_idx;

        } else {
            // 2. Knight hops (planes 56-63)
            bool found = false;
            for (int ki = 0; ki < 8; ++ki) {
                if (dc == KNIGHT_MOVES[ki].first && dr_agent == KNIGHT_MOVES[ki].second) {
                    plane = 56 + ki;
                    found = true;
                    break;
                }
            }

            // 3. Queen-like slides (planes 0-55), including queen promotions
            if (!found) {
                const int distance = std::max(std::abs(dr_agent), std::abs(dc));
                if (distance > 0) {
                    const int dir_dx = dc       / distance;
                    const int dir_dy = dr_agent / distance;
                    for (int di = 0; di < 8; ++di) {
                        if (dir_dx == DIRECTIONS[di].first && dir_dy == DIRECTIONS[di].second) {
                            plane = di * 7 + (distance - 1);
                            break;
                        }
                    }
                }
            }
        }

        if (plane >= 0 && plane < ACTION_PLANES) {
            mask[t_row * COLS * ACTION_PLANES + t_col * ACTION_PLANES + plane] = true;
        }
    }

    return mask;
}

// ── apply_action ─────────────────────────────────────────────────────────────
//
// Decodes action_idx to a move in the agent's coordinate frame, then executes
// it on the board.  Returns {true, move} on success, {false, NO_MOVE} on failure.
//
// Uses chess::uci::uciToMove() to reconstruct the full Move object (correctly
// handling en passant, castling, and promotions from board state).

std::pair<bool, chess::Move> ChessEnv::apply_action(int action_idx, chess::Board& board) {
    const bool flip = (board.sideToMove() == chess::Color::BLACK);

    // Decode action_idx as (row, col, move_type) — matches Python's np.unravel_index
    const int move_type = action_idx % ACTION_PLANES;
    const int tmp       = action_idx / ACTION_PLANES;
    const int col       = tmp % COLS;
    const int row       = tmp / COLS;

    // from_square: agent (row, col) → real board (file, rank)
    const int real_row        = flip ? (7 - row) : row;
    const int from_chess_rank = 7 - real_row;
    const int from_chess_file = col;
    const chess::Square from_sq(from_chess_file + from_chess_rank * 8);

    // to_square (agent coords) — determined by move_type
    int to_col = -1, to_row = -1;
    chess::PieceType promo = chess::PieceType::NONE;

    if (move_type < 56) {
        // Sliding move (queen/rook/bishop/king/pawn pushes)
        const int dir_idx  = move_type / 7;
        const int distance = (move_type % 7) + 1;
        const int dx = DIRECTIONS[dir_idx].first;
        const int dy = DIRECTIONS[dir_idx].second;

        to_col = col + dx * distance;
        to_row = row - dy * distance;  // agent-frame row (dy>0 = "north" = row decreases)

        if (to_col < 0 || to_col >= 8 || to_row < 0 || to_row >= 8)
            return {false, chess::Move::NO_MOVE};

        // Queen promotion: pawn reaching the back rank via a queen-like move
        if (board.at(from_sq).type() == chess::PieceType::PAWN) {
            const int r_to_row   = flip ? (7 - to_row) : to_row;
            const int to_ch_rank = 7 - r_to_row;
            if (to_ch_rank == 7 || to_ch_rank == 0)
                promo = chess::PieceType::QUEEN;
        }

    } else if (move_type < 64) {
        // Knight hop
        const int ki = move_type - 56;
        to_col = col + KNIGHT_MOVES[ki].first;
        to_row = row - KNIGHT_MOVES[ki].second;

        if (to_col < 0 || to_col >= 8 || to_row < 0 || to_row >= 8)
            return {false, chess::Move::NO_MOVE};

    } else {
        // Under-promotion
        const int promo_idx = move_type - 64;
        const int direction = promo_idx / 3;
        promo = UNDERPROMO_PIECES[promo_idx % 3];

        // Agent frame: pawn always moves "up" (dy = -1 → tensor_row - 1)
        const int dx = (direction == 0) ? 0 : (direction == 1) ? -1 : 1;
        to_col = col + dx;
        to_row = row - 1;  // row + dy where dy = -1

        if (to_col < 0 || to_col >= 8 || to_row < 0 || to_row >= 8)
            return {false, chess::Move::NO_MOVE};
    }

    // Convert to_col / to_row (agent) → real board rank/file
    const int real_to_row   = flip ? (7 - to_row) : to_row;
    const int to_chess_rank = 7 - real_to_row;
    const chess::Square to_sq(to_col + to_chess_rank * 8);

    // Build UCI string — uciToMove() auto-detects en passant, castling, promotion
    std::string uci;
    uci.reserve(5);
    uci += static_cast<char>('a' + static_cast<int>(from_sq.file()));
    uci += static_cast<char>('1' + static_cast<int>(from_sq.rank()));
    uci += static_cast<char>('a' + static_cast<int>(to_sq.file()));
    uci += static_cast<char>('1' + static_cast<int>(to_sq.rank()));
    if (promo != chess::PieceType::NONE) {
        constexpr char promo_chars[] = {'p', 'n', 'b', 'r', 'q', 'k'};
        uci += promo_chars[static_cast<int>(promo)];
    }

    chess::Move move = chess::uci::uciToMove(board, uci);
    if (move == chess::Move::NO_MOVE) return {false, chess::Move::NO_MOVE};
    if (!board.isLegal(move))         return {false, chess::Move::NO_MOVE};

    board.makeMove(move);
    return {true, move};
}

// ── get_evaluation ────────────────────────────────────────────────────────────
//
// Mirrors Python's ChessEnv.get_evaluation() exactly:
//   1. Material + PeSTO PST (tapered mg/eg by game phase)
//   2. Bishop pair bonus
//   3. Pawn structure (doubled, isolated, passed)
//   4. Rook bonuses (open/half-open file, 7th rank)
//   5. King safety (open files near king, pawn shelter)
//   6. Symmetric mobility
//   7. Normalise: win_prob = 2 / (1 + 10^(-cp/400)) - 1
//
// All piece-square tables are verbatim copies of the Python _PST arrays.

namespace {

// ── Piece-Square Tables (PeSTO) ──────────────────────────────────────────────
// Index = chess square: a1=0 .. h1=7, a2=8 .. h8=63.
// White uses index directly; Black uses chess::Square::flip() (mirror rank).

constexpr int PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr int PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr int KNIGHT_MG[64] = {
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
};
constexpr int KNIGHT_EG[64] = {
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};
constexpr int BISHOP_MG[64] = {
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};
constexpr int BISHOP_EG[64] = {
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};
constexpr int ROOK_MG[64] = {
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
};
constexpr int ROOK_EG[64] = {
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
};
constexpr int QUEEN_MG[64] = {
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};
constexpr int QUEEN_EG[64] = {
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};
constexpr int KING_MG[64] = {
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};
constexpr int KING_EG[64] = {
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
};

// Phase weights: 0=pawn,1=knight,1=bishop,2=rook,4=queen (king excluded)
// Total phase = 16*0 + 4*1 + 4*1 + 4*2 + 2*4 = 24
constexpr int TOTAL_PHASE = 24;

// PieceType is a wrapper class; compare with == rather than switch-case.
inline int piece_phase(chess::PieceType pt) {
    if (pt == chess::PieceType::KNIGHT || pt == chess::PieceType::BISHOP) return 1;
    if (pt == chess::PieceType::ROOK)  return 2;
    if (pt == chess::PieceType::QUEEN) return 4;
    return 0;
}

// Material values in centipawns (mg, eg) — match Python's _MAT_MG/_MAT_EG
inline std::pair<int,int> mat_val(chess::PieceType pt) {
    if (pt == chess::PieceType::PAWN)   return { 82,  94};
    if (pt == chess::PieceType::KNIGHT) return {337, 281};
    if (pt == chess::PieceType::BISHOP) return {365, 297};
    if (pt == chess::PieceType::ROOK)   return {477, 512};
    if (pt == chess::PieceType::QUEEN)  return {1025, 936};
    return {0, 0};
}

// PST lookup: index for White = sq.index() (a1=0..h8=63)
//             index for Black = mirror rank (same as Python's square_mirror)
inline int pst_idx(chess::Square sq, chess::Color color) {
    int raw = sq.index();
    if (color == chess::Color::BLACK) {
        int file = raw % 8;
        int rank = raw / 8;
        return (7 - rank) * 8 + file;
    }
    return raw;
}

inline std::pair<int,int> pst_val(chess::PieceType pt, int idx) {
    if (pt == chess::PieceType::PAWN)   return {PAWN_MG[idx],   PAWN_EG[idx]};
    if (pt == chess::PieceType::KNIGHT) return {KNIGHT_MG[idx], KNIGHT_EG[idx]};
    if (pt == chess::PieceType::BISHOP) return {BISHOP_MG[idx], BISHOP_EG[idx]};
    if (pt == chess::PieceType::ROOK)   return {ROOK_MG[idx],   ROOK_EG[idx]};
    if (pt == chess::PieceType::QUEEN)  return {QUEEN_MG[idx],  QUEEN_EG[idx]};
    if (pt == chess::PieceType::KING)   return {KING_MG[idx],   KING_EG[idx]};
    return {0, 0};
}

// Build file occupancy bitmask. Returns 8-bit value: bit f ↔ piece on file f.
// Uses Bitboard::pop() which removes and returns the lsb index.
inline uint8_t occupied_files(chess::Bitboard bb) {
    uint8_t files = 0;
    while (bb) {
        int sq = bb.pop();   // removes lsb, returns its index as int
        files |= static_cast<uint8_t>(1 << (sq % 8));
    }
    return files;
}

} // anonymous namespace

float ChessEnv::get_evaluation(const chess::Board& board) {
    // Terminal position: return exact reward
    auto [over, checkmate] = game_over(board);
    if (over) {
        if (checkmate) {
            // sideToMove is the mated player
            return (board.sideToMove() == chess::Color::BLACK) ? 1.0f : -1.0f;
        }
        return 0.0f;  // draw
    }

    int mg_score = 0;
    int eg_score = 0;
    int phase    = 0;

    // ── 1. MATERIAL + PST ────────────────────────────────────────────────────
    for (int sq_int = 0; sq_int < 64; ++sq_int) {
        chess::Square sq(sq_int);
        chess::Piece  piece = board.at(sq);
        if (piece == chess::Piece::NONE) continue;

        chess::PieceType pt    = piece.type();
        chess::Color     color = piece.color();
        int              sign  = (color == chess::Color::WHITE) ? 1 : -1;
        int              idx   = pst_idx(sq, color);

        auto [mat_mg, mat_eg] = mat_val(pt);
        auto [pst_mg, pst_eg] = pst_val(pt, idx);

        mg_score += sign * (mat_mg + pst_mg);
        eg_score += sign * (mat_eg + pst_eg);
        phase    += piece_phase(pt);
    }

    phase    = std::min(phase, TOTAL_PHASE);
    int cp   = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    // ── 2. BISHOP PAIR ───────────────────────────────────────────────────────
    {
        int wb = board.pieces(chess::PieceType::BISHOP, chess::Color::WHITE).count();
        int bb_cnt = board.pieces(chess::PieceType::BISHOP, chess::Color::BLACK).count();
        if (wb >= 2)    cp += 30;
        if (bb_cnt >= 2) cp -= 30;
    }

    // ── 3. PAWN STRUCTURE ────────────────────────────────────────────────────
    for (int ci = 0; ci < 2; ++ci) {
        chess::Color color = (ci == 0) ? chess::Color::WHITE : chess::Color::BLACK;
        chess::Color opp   = ~color;
        int          sign  = (ci == 0) ? 1 : -1;

        chess::Bitboard pawns     = board.pieces(chess::PieceType::PAWN, color);
        chess::Bitboard opp_pawns = board.pieces(chess::PieceType::PAWN, opp);

        // Build file presence arrays for friendly pawns
        uint8_t pawn_files_mask = occupied_files(pawns);

        chess::Bitboard tmp = pawns;
        while (tmp) {
            int  sq_i = tmp.pop();  // removes lsb, returns its index

            int f = sq_i % 8;
            int r = sq_i / 8;

            // Doubled: another friendly pawn on same file
            chess::Bitboard file_bb = chess::Bitboard(0x0101010101010101ULL << f);
            int pawns_on_file = (pawns & file_bb).count();
            if (pawns_on_file > 1) cp -= sign * 20;

            // Isolated: no friendly pawn on adjacent files
            uint8_t adj = 0;
            if (f > 0) adj |= static_cast<uint8_t>(1 << (f - 1));
            if (f < 7) adj |= static_cast<uint8_t>(1 << (f + 1));
            if (!(pawn_files_mask & adj)) cp -= sign * 20;

            // Passed pawn: no enemy pawn on same/adjacent files ahead of this pawn
            chess::Bitboard ahead_mask = 0;
            if (color == chess::Color::WHITE) {
                // squares strictly ahead (higher ranks) on f, f-1, f+1
                for (int rr = r + 1; rr <= 7; ++rr) {
                    ahead_mask |= chess::Bitboard(1ULL << (rr * 8 + f));
                    if (f > 0) ahead_mask |= chess::Bitboard(1ULL << (rr * 8 + f - 1));
                    if (f < 7) ahead_mask |= chess::Bitboard(1ULL << (rr * 8 + f + 1));
                }
            } else {
                for (int rr = 0; rr < r; ++rr) {
                    ahead_mask |= chess::Bitboard(1ULL << (rr * 8 + f));
                    if (f > 0) ahead_mask |= chess::Bitboard(1ULL << (rr * 8 + f - 1));
                    if (f < 7) ahead_mask |= chess::Bitboard(1ULL << (rr * 8 + f + 1));
                }
            }

            if (!(opp_pawns & ahead_mask)) {
                // Passed pawn bonus by rank (advance_rank = how far it has advanced)
                static constexpr int rank_bonus_w[8] = {0, 10, 20, 40, 60, 90, 120, 0};
                static constexpr int rank_bonus_b[8] = {0, 120, 90, 60, 40, 20, 10, 0};
                int advance_rank = (color == chess::Color::WHITE) ? r : (7 - r);
                int bonus = (color == chess::Color::WHITE) ? rank_bonus_w[advance_rank]
                                                           : rank_bonus_b[advance_rank];
                cp += sign * bonus;
            }
        }
    }

    // ── 4. ROOK BONUSES ──────────────────────────────────────────────────────
    for (int ci = 0; ci < 2; ++ci) {
        chess::Color color        = (ci == 0) ? chess::Color::WHITE : chess::Color::BLACK;
        chess::Color opp          = ~color;
        int          sign         = (ci == 0) ? 1 : -1;
        int          seventh_rank = (ci == 0) ? 6 : 1;

        chess::Bitboard own_pawns = board.pieces(chess::PieceType::PAWN, color);
        chess::Bitboard opp_pawns = board.pieces(chess::PieceType::PAWN, opp);
        chess::Bitboard rooks     = board.pieces(chess::PieceType::ROOK, color);

        chess::Bitboard tmp = rooks;
        while (tmp) {
            int  sq_i = tmp.pop();

            int f = sq_i % 8;
            int r = sq_i / 8;
            chess::Bitboard file_bb = chess::Bitboard(0x0101010101010101ULL << f);

            if (!(own_pawns & file_bb) && !(opp_pawns & file_bb)) {
                cp += sign * 20;  // open file
            } else if (!(own_pawns & file_bb)) {
                cp += sign * 10;  // half-open file
            }
            if (r == seventh_rank) cp += sign * 30;
        }
    }

    // ── 5. KING SAFETY ───────────────────────────────────────────────────────
    for (int ci = 0; ci < 2; ++ci) {
        chess::Color color = (ci == 0) ? chess::Color::WHITE : chess::Color::BLACK;
        chess::Color opp   = ~color;
        int          sign  = (ci == 0) ? 1 : -1;

        chess::Square king_sq = board.kingSq(color);
        int king_file = king_sq.index() % 8;
        int king_rank = king_sq.index() / 8;

        chess::Bitboard own_pawns = board.pieces(chess::PieceType::PAWN, color);
        chess::Bitboard opp_pawns = board.pieces(chess::PieceType::PAWN, opp);

        int f_lo = std::max(0, king_file - 1);
        int f_hi = std::min(7, king_file + 1);
        for (int f = f_lo; f <= f_hi; ++f) {
            chess::Bitboard file_bb = chess::Bitboard(0x0101010101010101ULL << f);
            if (!(own_pawns & file_bb)) {
                // No own pawn on this file near king
                bool fully_open = !(opp_pawns & file_bb);
                cp -= sign * (fully_open ? 50 : 25);
            }
        }

        // Pawn shelter: three squares directly in front of the king
        int shelter_rank = king_rank + (color == chess::Color::WHITE ? 1 : -1);
        if (shelter_rank >= 0 && shelter_rank <= 7) {
            for (int f = f_lo; f <= f_hi; ++f) {
                chess::Bitboard shelter_sq = chess::Bitboard(
                    1ULL << (shelter_rank * 8 + f));
                if (!(own_pawns & shelter_sq)) {
                    cp -= sign * 20;
                }
            }
        }
    }

    // ── 6. SYMMETRIC MOBILITY ────────────────────────────────────────────────
    // Mirror Python: count current side's legal moves, then flip the turn in a
    // copy to count the opponent's legal moves (Python bypasses check detection
    // by directly toggling board.turn — we do the equivalent via FEN substitution).
    {
        const bool white_to_move = (board.sideToMove() == chess::Color::WHITE);

        // Count current-side moves on the original board.
        chess::Movelist cur_moves;
        chess::movegen::legalmoves(cur_moves, board);

        // Build a copy with the opposite turn via FEN string substitution.
        std::string fen = board.getFen();
        // FEN format: "pieces turn castling ep halfmove fullmove"
        // Replace the turn token ('w' ↔ 'b') after the first space.
        auto turn_pos = fen.find(' ');
        if (turn_pos != std::string::npos && turn_pos + 1 < fen.size()) {
            fen[turn_pos + 1] = white_to_move ? 'b' : 'w';
        }
        chess::Board opp_board(fen);
        chess::Movelist opp_moves;
        chess::movegen::legalmoves(opp_moves, opp_board);

        int cur_mob = static_cast<int>(cur_moves.size());
        int opp_mob = static_cast<int>(opp_moves.size());
        // Always score White mobility positively.
        int white_mob = white_to_move ? cur_mob : opp_mob;
        int black_mob = white_to_move ? opp_mob : cur_mob;
        cp += 5 * (white_mob - black_mob);
    }

    // ── 7. NORMALIZE ─────────────────────────────────────────────────────────
    // win_prob = 2 / (1 + 10^(-cp/400)) - 1
    const float exponent  = -static_cast<float>(cp) / 400.0f;
    const float win_prob  = 2.0f / (1.0f + std::pow(10.0f, exponent)) - 1.0f;
    return win_prob;
}
