#!/usr/bin/env python3
"""
Chess RL GUI
PySide6 + python-chess SVG rendering.

Install dependency:  pip install pyside6
"""

import sys
import os
import random
from typing import Optional

import chess
import chess.svg
import numpy as np
import torch

from PySide6.QtCore    import Qt, QByteArray, QThread, Signal, QTimer
from PySide6.QtGui     import QColor, QFont, QPainter, QPen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication, QDialog, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QRadioButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from Agent      import MCTS, Node
from ChessEnv   import ChessEnv
from MCTS_simple import MCTS_simple

# ── Constants ─────────────────────────────────────────────────────────────────
BOARD_SIZE     = 600     # pixels
NUM_SIM_NEURAL = 800     # simulations per AI move (neural); divisible by LEAF_BATCH
NUM_SIM_SIMPLE = 5_000   # simulations per AI move (simple MCTS)
LEAF_BATCH     = 8       # must match the value used during training
TOP_N          = 5       # move rows shown in the probability panel


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_top_moves(root: Node, board: chess.Board, n: int = TOP_N) -> list:
    """Return up to n moves from root sorted by visit count."""
    if not root.is_expanded or root.legal_moves_cache is None:
        return []
    total = float(np.sum(root.N))
    if total == 0:
        return []
    legal  = np.where(root.legal_moves_cache)[0]
    order  = np.argsort(-root.N[legal])[:n]
    result = []
    for i in order:
        idx    = int(legal[i])
        visits = float(root.N[idx])
        if visits == 0:
            break
        b = board.copy()
        valid, move = ChessEnv.apply_action(idx, b)
        if valid:
            result.append({
                "uci":    move.uci(),
                "visits": int(visits),
                "pct":    visits / total * 100,
            })
    return result


def load_latest_model(agent: MCTS) -> Optional[str]:
    """
    Try to load the most recent model weights into agent.model.
    Preference order: checkpoint_latest.pt > model_ep*.pt > model_checkpoint*.pt
    Returns the filename on success, None if nothing is found / all fail.
    """
    candidates: list[str] = []
    try:
        for name in sorted(os.listdir(".")):
            if name == "checkpoint_latest.pt":
                candidates.insert(0, name)
            elif name.startswith("model_ep") or name.startswith("model_checkpoint"):
                candidates.append(name)
    except OSError:
        return None

    for fname in candidates:
        try:
            ckpt = torch.load(fname, map_location=agent.device)
            state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
            agent.model.load_state_dict(state)
            agent.model.eval()
            return fname
        except Exception:
            continue
    return None


# ── Eval Bar ──────────────────────────────────────────────────────────────────

class EvalBar(QWidget):
    """
    Thin vertical bar: black fills from the top, white fills from the bottom.
    value in [-1, 1]:  +1 = white winning,  -1 = black winning.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._value: float = 0.0
        self.setFixedWidth(22)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Position evaluation  (W = white advantage, B = black advantage)")

    def set_value(self, v: float) -> None:
        self._value = max(-1.0, min(1.0, float(v)))
        self.update()

    def paintEvent(self, _event) -> None:
        p   = QPainter(self)
        w   = self.width()
        h   = self.height()

        # Background = black player's colour
        p.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # White portion grows from bottom
        white_h = int((self._value + 1.0) / 2.0 * h)
        p.fillRect(0, h - white_h, w, white_h, QColor(225, 225, 225))

        # Centre divider
        p.setPen(QPen(QColor(110, 110, 110), 1))
        p.drawLine(0, h // 2, w, h // 2)

        # Labels
        font = QFont("Arial", 7, QFont.Bold)
        p.setFont(font)
        p.setPen(QColor(200, 200, 200))
        p.drawText(0, 2, w, 14, Qt.AlignHCenter, "B")
        p.setPen(QColor(80, 80, 80))
        p.drawText(0, h - 14, w, 14, Qt.AlignHCenter, "W")


# ── Move Probability Panel ────────────────────────────────────────────────────

class _MoveRow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(4)

        self.lbl_move = QLabel("----")
        self.lbl_move.setFixedWidth(46)
        self.lbl_move.setFont(QFont("Courier", 9, QFont.Bold))

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(13)
        self.bar.setStyleSheet(
            "QProgressBar { background:#ddd; border-radius:2px; }"
            "QProgressBar::chunk { background:#4a90d9; border-radius:2px; }"
        )

        self.lbl_pct = QLabel("  0%")
        self.lbl_pct.setFixedWidth(32)
        self.lbl_pct.setFont(QFont("Courier", 9))
        self.lbl_pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self.lbl_move)
        layout.addWidget(self.bar)
        layout.addWidget(self.lbl_pct)

    def set_data(self, uci: str, pct: float) -> None:
        self.lbl_move.setText(uci)
        self.bar.setValue(int(pct))
        self.lbl_pct.setText(f"{pct:3.0f}%")


class MoveProbPanel(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Top Moves", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        self._rows = [_MoveRow() for _ in range(TOP_N)]
        for row in self._rows:
            layout.addWidget(row)
        self.clear()

    def update_moves(self, moves: list) -> None:
        for i, row in enumerate(self._rows):
            if i < len(moves):
                row.set_data(moves[i]["uci"], moves[i]["pct"])
                row.setVisible(True)
            else:
                row.setVisible(False)

    def clear(self) -> None:
        for row in self._rows:
            row.setVisible(False)


# ── Chess Board Widget ────────────────────────────────────────────────────────

class ChessBoardWidget(QSvgWidget):
    """
    Renders the board using chess.svg.  No image files needed.

    Click interaction:
      • First click  — select a friendly piece; legal destinations are dotted.
      • Second click — move to a destination (queen-promotion auto-applied);
                       clicking another friendly piece re-selects instead.

    Coordinate mapping (coordinates=False ⇒ board fills the entire SVG):
      White orientation:  file = col,     rank = 7 − row
      Black orientation:  file = 7 − col, rank = row
    """

    human_move = Signal(chess.Move)

    def __init__(
        self,
        human_color: chess.Color = chess.WHITE,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.human_color  = human_color
        self.board        = chess.Board()
        self.last_move:   Optional[chess.Move]   = None
        self.selected_sq: Optional[chess.Square] = None
        self.legal_dests: set[chess.Square]       = set()
        self.arrows:      list                    = []
        self.interactive: bool                    = True

        self.setFixedSize(BOARD_SIZE, BOARD_SIZE)
        self.update_board()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def update_board(self) -> None:
        orientation = chess.WHITE if self.human_color == chess.WHITE else chess.BLACK

        fill: dict[chess.Square, str] = {}
        if self.selected_sq is not None:
            fill[self.selected_sq] = "#7fc97f"      # green selection tint

        svg = chess.svg.board(
            self.board,
            orientation  = orientation,
            lastmove     = self.last_move,
            fill         = fill,
            squares      = chess.SquareSet(self.legal_dests),
            arrows       = self.arrows,
            size         = BOARD_SIZE,
            coordinates  = False,   # no border labels → no margin offset needed
        )
        self.load(QByteArray(svg.encode("utf-8")))

    # ── Input handling ────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if not self.interactive or event.button() != Qt.LeftButton:
            return
        if self.board.turn != self.human_color or self.board.is_game_over():
            return

        sq = self._pixel_to_square(event.position().x(), event.position().y())

        if self.selected_sq is None:
            # ── Select piece ───────────────────────────────────────────────
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.human_color:
                self.selected_sq = sq
                self.legal_dests = {
                    m.to_square for m in self.board.legal_moves
                    if m.from_square == sq
                }
                self.update_board()
        else:
            # ── Attempt move ───────────────────────────────────────────────
            move  = chess.Move(self.selected_sq, sq)
            promo = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)

            made: Optional[chess.Move] = None
            if   move  in self.board.legal_moves: made = move
            elif promo in self.board.legal_moves: made = promo

            if made is not None:
                self.selected_sq = None
                self.legal_dests = set()
                self.arrows      = []
                self.human_move.emit(made)
            else:
                # Re-select a different piece or deselect
                piece = self.board.piece_at(sq)
                if piece and piece.color == self.human_color:
                    self.selected_sq = sq
                    self.legal_dests = {
                        m.to_square for m in self.board.legal_moves
                        if m.from_square == sq
                    }
                else:
                    self.selected_sq = None
                    self.legal_dests = set()
                self.update_board()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pixel_to_square(self, px: float, py: float) -> chess.Square:
        w   = self.width()
        h   = self.height()
        col = max(0, min(7, int(px * 8 / w)))
        row = max(0, min(7, int(py * 8 / h)))
        if self.human_color == chess.WHITE:
            file_, rank_ = col, 7 - row
        else:
            file_, rank_ = 7 - col, row
        return chess.square(file_, rank_)


# ── AI Worker Thread ──────────────────────────────────────────────────────────

class AIWorker(QThread):
    """
    Runs MCTS in a background thread so the UI never freezes.

    Signals
    -------
    progress(list)       — emitted periodically during neural search;
                           each item: {"uci": str, "visits": int, "pct": float}
    move_ready(object)   — emitted once with the chosen chess.Move (or None)
    """

    progress   = Signal(list)
    move_ready = Signal(object)

    def __init__(
        self,
        board:              chess.Board,
        ai_type:            str,
        model_state_dict:   Optional[dict] = None,
        mcts_simple_agent:  Optional[MCTS_simple] = None,
        num_sim:            int  = NUM_SIM_NEURAL,
        leaf_batch:         int  = LEAF_BATCH,
    ) -> None:
        super().__init__()
        self.board             = board.copy()
        self.ai_type           = ai_type
        self.model_state_dict  = model_state_dict
        self.mcts_simple_agent = mcts_simple_agent
        self.num_sim           = num_sim
        self.leaf_batch        = leaf_batch
        self._cancel           = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        if self.ai_type == "neural":
            self._run_neural()
        else:
            self._run_simple()

    # ── Neural ────────────────────────────────────────────────────────────────

    def _run_neural(self) -> None:
        agent = MCTS(
            num_simulations = self.num_sim,
            leaf_batch_size = self.leaf_batch,
            buffer_size     = 1,
        )
        if self.model_state_dict:
            agent.model.load_state_dict(self.model_state_dict)
        agent.model.eval()

        state = ChessEnv.encode_state(self.board)
        root  = Node(state, self.board.copy())

        n_batches    = self.num_sim // self.leaf_batch
        report_every = max(1, n_batches // 10)   # ~10 progress signals total

        for i in range(n_batches):
            if self._cancel:
                return
            agent.run_simulation_batch(root)
            if (i + 1) % report_every == 0:
                self.progress.emit(get_top_moves(root, self.board))

        if self._cancel:
            return

        if np.sum(root.N) == 0:
            legal = list(self.board.legal_moves)
            self.move_ready.emit(random.choice(legal) if legal else None)
            return

        action_idx = int(np.argmax(root.N))
        b = self.board.copy()
        valid, move = ChessEnv.apply_action(action_idx, b)

        # Final update so UI shows settled probabilities
        self.progress.emit(get_top_moves(root, self.board))
        self.move_ready.emit(move if valid else (
            random.choice(list(self.board.legal_moves))
            if self.board.legal_moves else None
        ))

    # ── Simple ────────────────────────────────────────────────────────────────

    def _run_simple(self) -> None:
        move = self.mcts_simple_agent.best_move(self.board)
        if not self._cancel:
            self.move_ready.emit(move)


# ── Setup Dialog ──────────────────────────────────────────────────────────────

class SetupDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chess RL")
        self.setModal(True)
        self.setMinimumWidth(340)

        self.ai_type:     str          = "neural"
        self.human_color: chess.Color  = chess.WHITE

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(28, 24, 28, 24)

        # Title
        lbl_title = QLabel("Chess RL")
        lbl_title.setFont(QFont("Arial", 22, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_sub = QLabel("AlphaZero-style reinforcement learning")
        lbl_sub.setFont(QFont("Arial", 9))
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setStyleSheet("color:#888;")
        layout.addWidget(lbl_sub)

        # Opponent type
        ai_box    = QGroupBox("Opponent")
        ai_layout = QVBoxLayout(ai_box)
        self.radio_neural = QRadioButton("Neural MCTS  (uses trained model)")
        self.radio_simple = QRadioButton("Simple MCTS  (pure tree search)")
        self.radio_neural.setChecked(True)
        ai_layout.addWidget(self.radio_neural)
        ai_layout.addWidget(self.radio_simple)
        layout.addWidget(ai_box)

        # Colour choice
        color_box    = QGroupBox("Play as")
        color_layout = QHBoxLayout(color_box)
        self.radio_white = QRadioButton("White")
        self.radio_black = QRadioButton("Black")
        self.radio_white.setChecked(True)
        color_layout.addWidget(self.radio_white)
        color_layout.addWidget(self.radio_black)
        layout.addWidget(color_box)

        btn = QPushButton("Start Game")
        btn.setFont(QFont("Arial", 11))
        btn.setFixedHeight(38)
        btn.clicked.connect(self._on_start)
        layout.addWidget(btn)

    def _on_start(self) -> None:
        self.ai_type     = "neural" if self.radio_neural.isChecked() else "simple"
        self.human_color = chess.WHITE if self.radio_white.isChecked() else chess.BLACK
        self.accept()


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, ai_type: str, human_color: chess.Color) -> None:
        super().__init__()
        self.ai_type     = ai_type
        self.human_color = human_color
        self.board       = chess.Board()
        self.ai_worker:  Optional[AIWorker] = None

        self.mcts_agent:   Optional[MCTS]        = None
        self.simple_agent: Optional[MCTS_simple] = None
        self._model_info   = self._init_ai()
        self._init_ui()

        # If human plays black the AI must move first — defer until window is shown
        QTimer.singleShot(200, self._maybe_trigger_ai)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_ai(self) -> str:
        if self.ai_type == "neural":
            self.mcts_agent = MCTS(
                buffer_size     = 1,
                num_simulations = NUM_SIM_NEURAL,
                leaf_batch_size = LEAF_BATCH,
            )
            fname = load_latest_model(self.mcts_agent)
            return f"Loaded {fname}" if fname else "No checkpoint found — untrained model"
        else:
            self.simple_agent = MCTS_simple(num_simulations=NUM_SIM_SIMPLE)
            return f"Simple MCTS  ({NUM_SIM_SIMPLE:,} simulations)"

    def _init_ui(self) -> None:
        self.setWindowTitle("Chess RL")

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # ── Left column: eval bar + board ────────────────────────────────────
        left = QHBoxLayout()
        left.setSpacing(6)

        self.eval_bar = EvalBar()
        left.addWidget(self.eval_bar)

        self.board_wgt = ChessBoardWidget(self.human_color)
        self.board_wgt.human_move.connect(self._on_human_move)
        left.addWidget(self.board_wgt)

        root_layout.addLayout(left)

        # ── Right column: info panel ──────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(235)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(8)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        # Status group
        status_box    = QGroupBox("Game Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.setSpacing(4)

        self.lbl_status = QLabel("White to move")
        self.lbl_status.setFont(QFont("Arial", 11, QFont.Bold))
        self.lbl_status.setWordWrap(True)

        color_str = "White ♔" if self.human_color == chess.WHITE else "Black ♚"
        self.lbl_color = QLabel(f"You: {color_str}")
        self.lbl_color.setFont(QFont("Arial", 9))

        self.lbl_ai = QLabel(f"AI:  {self.ai_type.capitalize()} MCTS")
        self.lbl_ai.setFont(QFont("Arial", 9))

        self.lbl_model = QLabel(self._model_info)
        self.lbl_model.setFont(QFont("Arial", 8))
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet("color:#666;")

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_color)
        status_layout.addWidget(self.lbl_ai)
        status_layout.addWidget(self.lbl_model)
        panel_layout.addWidget(status_box)

        # Thinking indicator
        self.lbl_thinking = QLabel("")
        self.lbl_thinking.setFont(QFont("Arial", 9))
        self.lbl_thinking.setStyleSheet("color:#555; font-style:italic;")
        self.lbl_thinking.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.lbl_thinking)

        # Move probability panel (only meaningful for neural MCTS)
        self.move_prob = MoveProbPanel()
        panel_layout.addWidget(self.move_prob)

        panel_layout.addStretch()

        # Controls
        ctrl_box    = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)
        btn_new     = QPushButton("New Game  [R]")
        btn_new.clicked.connect(self.reset_game)
        ctrl_layout.addWidget(btn_new)
        panel_layout.addWidget(ctrl_box)

        root_layout.addWidget(panel)
        self.adjustSize()

    # ── Game logic ────────────────────────────────────────────────────────────

    def _on_human_move(self, move: chess.Move) -> None:
        self.board_wgt.arrows = []          # clear any AI arrows from previous turn
        self.board.push(move)
        self.board_wgt.board     = self.board
        self.board_wgt.last_move = move
        self.board_wgt.update_board()
        self._update_eval()
        self._update_status()
        if not self.board.is_game_over():
            self._trigger_ai()

    def _maybe_trigger_ai(self) -> None:
        """Trigger AI if it is its turn (e.g. human plays black at game start)."""
        if self.board.turn != self.human_color and not self.board.is_game_over():
            self._trigger_ai()

    def _trigger_ai(self) -> None:
        if self.ai_worker and self.ai_worker.isRunning():
            return
        self.board_wgt.interactive = False
        self.lbl_thinking.setText("Thinking…")
        self.move_prob.clear()

        if self.ai_type == "neural":
            # Pass a CPU snapshot of the weights — safe to read from any thread
            sd = {k: v.cpu() for k, v in self.mcts_agent.model.state_dict().items()}
            self.ai_worker = AIWorker(
                board            = self.board,
                ai_type          = "neural",
                model_state_dict = sd,
                num_sim          = NUM_SIM_NEURAL,
                leaf_batch       = LEAF_BATCH,
            )
        else:
            self.ai_worker = AIWorker(
                board             = self.board,
                ai_type           = "simple",
                mcts_simple_agent = self.simple_agent,
                num_sim           = NUM_SIM_SIMPLE,
            )

        self.ai_worker.progress.connect(self._on_ai_progress)
        self.ai_worker.move_ready.connect(self._on_ai_move)
        # _on_worker_done is called only after the thread has fully stopped,
        # so it is safe to drop the Python reference there.
        self.ai_worker.finished.connect(self._on_worker_done)
        self.ai_worker.start()

    def _on_ai_progress(self, top_moves: list) -> None:
        """Called periodically while the neural AI is thinking."""
        self.move_prob.update_moves(top_moves)

        # Overlay top-3 moves as arrows on the board (green / orange / blue)
        arrow_colors = ["#15781B", "#B7450A", "#0044cc"]
        arrows = []
        for i, m in enumerate(top_moves[:3]):
            try:
                mv = chess.Move.from_uci(m["uci"])
                arrows.append(
                    chess.svg.Arrow(mv.from_square, mv.to_square, color=arrow_colors[i])
                )
            except Exception:
                pass
        self.board_wgt.arrows = arrows
        self.board_wgt.update_board()

    def _on_worker_done(self) -> None:
        """Called by the finished signal — the thread has fully stopped here."""
        self.ai_worker = None

    def _on_ai_move(self, move: Optional[chess.Move]) -> None:
        self.board_wgt.interactive = True
        self.lbl_thinking.setText("")
        self.board_wgt.arrows = []
        # Do NOT null ai_worker here: the thread may still be in Qt's cleanup
        # phase.  _on_worker_done (connected to finished) handles that.

        if move is None or move not in self.board.legal_moves:
            # Fallback: should rarely happen
            legal = list(self.board.legal_moves)
            if not legal:
                self._update_status()
                return
            move = random.choice(legal)

        self.board.push(move)
        self.board_wgt.board     = self.board
        self.board_wgt.last_move = move
        self.board_wgt.update_board()
        self._update_eval()
        self._update_status()

    # ── UI updates ────────────────────────────────────────────────────────────

    def _update_eval(self) -> None:
        if self.board.is_game_over():
            res = self.board.result()
            self.eval_bar.set_value(1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0)
        else:
            self.eval_bar.set_value(ChessEnv.get_evaluation(self.board))

    def _update_status(self) -> None:
        b = self.board
        if b.is_checkmate():
            winner = "Black" if b.turn == chess.WHITE else "White"
            self.lbl_status.setText(f"Checkmate — {winner} wins!")
            self.lbl_status.setStyleSheet("color:#c0392b; font-weight:bold;")
        elif b.is_stalemate():
            self.lbl_status.setText("Stalemate — Draw")
            self.lbl_status.setStyleSheet("color:#7f8c8d;")
        elif b.is_insufficient_material():
            self.lbl_status.setText("Insufficient material — Draw")
            self.lbl_status.setStyleSheet("color:#7f8c8d;")
        elif b.is_seventyfive_moves():
            self.lbl_status.setText("75-move rule — Draw")
            self.lbl_status.setStyleSheet("color:#7f8c8d;")
        elif b.is_fivefold_repetition():
            self.lbl_status.setText("Fivefold repetition — Draw")
            self.lbl_status.setStyleSheet("color:#7f8c8d;")
        elif b.is_game_over():
            self.lbl_status.setText("Game Over — Draw")
            self.lbl_status.setStyleSheet("color:#7f8c8d;")
        elif b.is_check():
            turn = "White" if b.turn == chess.WHITE else "Black"
            self.lbl_status.setText(f"Check! — {turn} to move")
            self.lbl_status.setStyleSheet("color:#e67e22; font-weight:bold;")
        else:
            turn = "White" if b.turn == chess.WHITE else "Black"
            self.lbl_status.setText(f"{turn} to move")
            self.lbl_status.setStyleSheet("")

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_game(self) -> None:
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.cancel()
            if not self.ai_worker.wait(2000):
                self.ai_worker.terminate()
                self.ai_worker.wait()
        if self.ai_worker:
            # Block any queued move_ready / progress signals so they cannot
            # fire against the new board after the reset.
            self.ai_worker.blockSignals(True)
        self.ai_worker = None

        self.board                 = chess.Board()
        self.board_wgt.board       = self.board
        self.board_wgt.last_move   = None
        self.board_wgt.selected_sq = None
        self.board_wgt.legal_dests = set()
        self.board_wgt.arrows      = []
        self.board_wgt.interactive = True
        self.board_wgt.update_board()
        self.move_prob.clear()
        self.lbl_thinking.setText("")
        self.lbl_status.setStyleSheet("")
        self.eval_bar.set_value(0.0)
        self._update_status()
        QTimer.singleShot(100, self._maybe_trigger_ai)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if   event.key() == Qt.Key_R:      self.reset_game()
        elif event.key() == Qt.Key_Escape: self.close()

    def closeEvent(self, event) -> None:
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.cancel()
            if not self.ai_worker.wait(2000):
                self.ai_worker.terminate()
                self.ai_worker.wait()
        event.accept()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")      # consistent look across Windows / macOS / Linux

    dialog = SetupDialog()
    if dialog.exec() != QDialog.Accepted:
        sys.exit(0)

    window = MainWindow(dialog.ai_type, dialog.human_color)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
