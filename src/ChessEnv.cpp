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

void ChessEnv::encode_state_into(const chess::Board& board, float* data) {
    constexpr int N = STATE_CHANNELS * BOARD_SZ * BOARD_SZ;
    std::fill(data, data + N, 0.0f);

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
}

torch::Tensor ChessEnv::encode_state(const chess::Board& board) {
    auto state = torch::zeros({STATE_CHANNELS, BOARD_SZ, BOARD_SZ}, torch::kFloat32);
    encode_state_into(board, state.data_ptr<float>());
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
