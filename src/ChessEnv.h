#pragma once
// ChessEnv.h — C++ mirror of ChessEnv.py
//
// Must produce byte-identical outputs to the Python versions of:
//   encode_state()      →  (20, 8, 8) float32 tensor
//   get_action_mask()   →  bool[4672] legal-move mask
//   apply_action()      →  apply action index to board, return {valid, move}
//
// Phase 2 task: implement these three functions and validate against golden
// vectors produced by generate_test_vectors.py.

#include <array>
#include <utility>
#include <torch/torch.h>
#include "chess.hpp"

// ── Action-space dimensions (must match Python) ───────────────────────────────
constexpr int ROWS          = 8;
constexpr int COLS          = 8;
constexpr int ACTION_PLANES = 73;    // 56 queen-like + 8 knight + 9 under-promo
constexpr int ACTION_DIM    = ROWS * COLS * ACTION_PLANES;  // 4672

// ── State dimensions ──────────────────────────────────────────────────────────
constexpr int STATE_CHANNELS = 20;
constexpr int BOARD_SZ       = 8;

// ── Move-encoding tables — defined in ChessEnv.cpp ───────────────────────────
// Matching Python's DIRECTIONS and KNIGHT_MOVES exactly.
// Each entry is {dx, dy} where dx = file-delta, dy = rank-delta in agent coords.

// Queen/rook/bishop sliding moves — 8 directions, up to 7 squares each (planes 0-55)
extern const std::array<std::pair<int,int>, 8> DIRECTIONS;

// Knight hops — 8 hops (planes 56-63)
extern const std::array<std::pair<int,int>, 8> KNIGHT_MOVES;

// Under-promotion piece types in plane order: ROOK, BISHOP, KNIGHT (planes 64-72)
// (Queen promotion is handled as a queen-like sliding move — no separate planes)
extern const std::array<chess::PieceType, 3> UNDERPROMO_PIECES;

// ── Type aliases ──────────────────────────────────────────────────────────────
using ActionMask = std::array<bool, ACTION_DIM>;
using StateArray = std::array<float, STATE_CHANNELS * BOARD_SZ * BOARD_SZ>; // 1280

// ── ChessEnv namespace ────────────────────────────────────────────────────────
namespace ChessEnv {

    // Encode a board position into a (STATE_CHANNELS, BOARD_SZ, BOARD_SZ) tensor.
    // The agent always sees itself as White:
    //   - When Black is to move, the board is flipped vertically and colors swapped.
    // Channels:
    //   0-5:  current-player pieces  (P N B R Q K)
    //   6-11: opponent pieces        (P N B R Q K)
    //   12:   turn plane             (always 1.0)
    //   13:   current-player kingside castling right
    //   14:   current-player queenside castling right
    //   15:   opponent kingside castling right
    //   16:   opponent queenside castling right
    //   17:   en-passant square
    //   18:   half-move clock / 50.0
    //   19:   repetition count / 2.0  (0, 0.5 or 1.0)
    torch::Tensor encode_state(const chess::Board& board);

    // Encode directly into a caller-supplied float buffer (STATE_CHANNELS*8*8 = 1280 floats).
    // Avoids heap allocation — safe to call from any worker thread without touching
    // PyTorch's global allocator or CUDA state.
    void encode_state_into(const chess::Board& board, float* out);

    // Returns a flat bool[4672] mask of legal moves in the agent's action space.
    // Indexing: mask[row * COLS * ACTION_PLANES + col * ACTION_PLANES + plane]
    ActionMask get_action_mask(const chess::Board& board);

    // Decode action_idx → chess move and push it onto the board.
    // Returns {true, move} on success, {false, chess::Move::NO_MOVE} on failure.
    std::pair<bool, chess::Move> apply_action(int action_idx, chess::Board& board);

    // Helper: is the game over?  Returns {is_over, is_checkmate}.
    std::pair<bool, bool> game_over(const chess::Board& board);

    // Tapered evaluation from White's perspective, normalized to [-1, 1].
    // Mirrors Python's ChessEnv.get_evaluation():
    //   - Tapered material + PeSTO piece-square tables (mg/eg interpolated by phase)
    //   - Bishop pair bonus
    //   - Pawn structure: doubled, isolated, passed pawns
    //   - Rook bonuses: open/half-open file, 7th rank
    //   - King safety: open files near king, pawn shelter
    //   - Symmetric mobility
    // Returns the terminal reward for game-over positions.
    float get_evaluation(const chess::Board& board);

} // namespace ChessEnv
