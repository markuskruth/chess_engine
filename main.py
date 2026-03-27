import pygame
import sys
import chess
import torch
import numpy as np
from Agent import MCTS
from ChessEnv import ChessEnv
from Neuralnet import CNNNet

START_FEN = "4k3/7R/K7/R7/8/8/8/8 w - - 0 1"

pygame.init()

# Display
display_size = 800
square_size = display_size // 8
win = pygame.display.set_mode((display_size, display_size))
pygame.display.set_caption("Chess RL - Play Against Trained Model")

img_path = "images/"

# Load images
pieces = {
    "P": pygame.image.load(f"{img_path}whitePawn.png"),
    "R": pygame.image.load(f"{img_path}whiteRook.png"),
    "N": pygame.image.load(f"{img_path}whiteKnight.png"),
    "B": pygame.image.load(f"{img_path}whiteBishop.png"),
    "Q": pygame.image.load(f"{img_path}whiteQueen.png"),
    "K": pygame.image.load(f"{img_path}whiteKing.png"),
    "p": pygame.image.load(f"{img_path}blackPawn.png"),
    "r": pygame.image.load(f"{img_path}blackRook.png"),
    "n": pygame.image.load(f"{img_path}blackKnight.png"),
    "b": pygame.image.load(f"{img_path}blackBishop.png"),
    "q": pygame.image.load(f"{img_path}blackQueen.png"),
    "k": pygame.image.load(f"{img_path}blackKing.png"),
}

# Scale images
for key in pieces:
    pieces[key] = pygame.transform.scale(pieces[key], (square_size, square_size))

# Global variables
ai_agent = None
human_color = chess.WHITE
game_over = False
message = ""
last_move = None


# Convert mouse position to chess square
def get_square_from_mouse(pos):
    x, y = pos
    file = x // square_size
    rank = 7 - (y // square_size)  # flip board vertically
    return chess.square(file, rank)


def initialize_game():
    """Show menu to select player color and initialize AI."""
    global human_color, ai_agent, game_over, message
    
    print("Initializing AI agent...")
    try:
        ai_agent = MCTS(buffer_size=100000)
        # Try to load a trained model if it exists
        import os
        model_files = [f for f in os.listdir('.') if f.startswith('model_checkpoint')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            print(f"Loading trained model: {latest_model}")
            ai_agent.model.load_state_dict(torch.load(latest_model, map_location=ai_agent.device))
            ai_agent.model.eval()
            message = f"Loaded {latest_model}"
        else:
            print("No trained model found. Using untrained model.")
            message = "Using untrained model"
    except Exception as e:
        print(f"Error initializing AI: {e}")
        message = f"Error loading model: {e}"
        return
    
    game_over = False


def draw_menu():
    """Draw color selection menu."""
    font = pygame.font.Font(None, 36)
    title = font.render("Choose Your Color", True, (0, 0, 0))
    white_text = font.render("Press W for White", True, (200, 200, 200))
    black_text = font.render("Press B for Black", True, (50, 50, 50))
    
    win.fill((240, 217, 181))
    win.blit(title, (display_size // 2 - title.get_width() // 2, 150))
    win.blit(white_text, (display_size // 2 - white_text.get_width() // 2, 300))
    win.blit(black_text, (display_size // 2 - black_text.get_width() // 2, 400))
    pygame.display.update()


def ai_make_move(board):
    """Use AI to select and make a move."""
    global message
    
    try:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return
        
        # Run fewer simulations for faster gameplay
        NUM_SIMULATIONS = 20
        state = ChessEnv.encode_state(board)
        root = ai_agent.memory.__class__.__bases__[0].__dict__.get('Node')
        
        # Simplified AI: use the model's policy output to select move
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ai_agent.device)
        with torch.no_grad():
            p, v = ai_agent.model(state_tensor)
        
        # Flatten and mask illegal moves
        p = p.cpu().numpy().flatten()
        action_mask = ChessEnv.get_action_mask(state).reshape(-1)
        p = np.where(action_mask, p, -np.inf)
        
        # Select best legal move
        if np.any(np.isfinite(p)):
            action = np.argmax(p)
            move = legal_moves[min(action % len(legal_moves), len(legal_moves) - 1)]
        else:
            # Fallback: random legal move
            move = legal_moves[np.random.randint(len(legal_moves))]
        
        board.push(move)
        color_name = "White" if board.turn == chess.BLACK else "Black"
        message = f"AI ({color_name}) played: {move.uci()}"
        
    except Exception as e:
        print(f"Error in AI move: {e}")
        message = f"AI Error: {e}"
        # Fallback: make random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(legal_moves[0])


def get_game_status(board):
    """Get current game status message."""
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate! {winner} wins!"
    elif board.is_stalemate():
        return "Stalemate!"
    elif board.is_check():
        return "Check!"
    elif board.is_game_over():
        return "Game Over - Draw"
    else:
        turn = "White" if board.turn == chess.WHITE else "Black"
        return f"{turn} to move"


# Draw board and pieces
def render(board):
    colors = [(240, 217, 181), (181, 136, 99)]

    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(
                win,
                color,
                pygame.Rect(
                    file * square_size,
                    (7 - rank) * square_size,
                    square_size,
                    square_size,
                ),
            )

            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece:
                symbol = piece.symbol()
                win.blit(
                    pieces[symbol],
                    (file * square_size, (7 - rank) * square_size),
                )

    # Display game status
    font = pygame.font.Font(None, 24)
    status_text = font.render(get_game_status(board), True, (0, 0, 0))
    message_text = font.render(message, True, (50, 50, 150))
    color_text = font.render(
        f"You are playing as {'White' if human_color == chess.WHITE else 'Black'}",
        True,
        (0, 100, 0)
    )

    pygame.draw.rect(win, (200, 200, 200), (0, 0, display_size, 60))
    win.blit(status_text, (10, 5))
    win.blit(color_text, (10, 25))
    win.blit(message_text, (300, 15))

    pygame.display.update()


# Handle move dragging
def handle_move(board):
    mouse_down = True
    start_pos = pygame.mouse.get_pos()
    from_sq = get_square_from_mouse(start_pos)

    piece = board.piece_at(from_sq)
    if piece is None or piece.color != board.turn:
        return False

    legal_moves = [
        move for move in board.legal_moves if move.from_square == from_sq
    ]

    while mouse_down:
        pygame.event.pump()
        mouse_down = pygame.mouse.get_pressed()[0]
        mouse_pos = pygame.mouse.get_pos()

        render(board)

        # Draw legal move indicators
        for move in legal_moves:
            to_sq = move.to_square
            file = chess.square_file(to_sq)
            rank = chess.square_rank(to_sq)
            pygame.draw.circle(
                win,
                (0, 255, 0),
                (
                    file * square_size + square_size // 2,
                    (7 - rank) * square_size + square_size // 2,
                ),
                10,
            )

        # Draw dragged piece
        if piece:
            win.blit(
                pieces[piece.symbol()],
                (mouse_pos[0] - square_size // 2, mouse_pos[1] - square_size // 2),
            )

        pygame.display.update()

    # Determine destination when mouse released
    end_pos = pygame.mouse.get_pos()
    to_sq = get_square_from_mouse(end_pos)

    move = chess.Move(from_sq, to_sq)

    # Handle promotion automatically (queen)
    if move in board.legal_moves:
        board.push(move)
        return True
    else:
        # try promotion
        move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        if move in board.legal_moves:
            board.push(move)
            return True
    
    return False


# MAIN LOOP
if __name__ == "__main__":
    # Color selection
    color_selected = False
    while not color_selected:
        draw_menu()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    human_color = chess.WHITE
                    color_selected = True
                elif event.key == pygame.K_b:
                    human_color = chess.BLACK
                    color_selected = True
    
    # Initialize game and AI
    initialize_game()
    
    board = chess.Board(START_FEN)
    
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(30)  # 30 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Only handle moves for human player
                if board.turn == human_color and not board.is_game_over():
                    if handle_move(board):
                        message = "Your move played"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset game
                    board = chess.Board("8/2ppk3/8/8/8/8/PPP1K3/8 w - - 0 1")
                    message = "Game reset"
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # AI makes move if it's AI's turn and game is not over
        if board.turn != human_color and not board.is_game_over():
            ai_make_move(board)

        render(board)