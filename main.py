import pygame
import sys
import chess

pygame.init()

# Display
display_size = 800
square_size = display_size // 8
win = pygame.display.set_mode((display_size, display_size))

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


# Convert mouse position to chess square
def get_square_from_mouse(pos):
    x, y = pos
    file = x // square_size
    rank = 7 - (y // square_size)  # flip board vertically
    return chess.square(file, rank)


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

    pygame.display.update()


# Handle move dragging
def handle_move(board):
    mouse_down = True
    start_pos = pygame.mouse.get_pos()
    from_sq = get_square_from_mouse(start_pos)

    piece = board.piece_at(from_sq)
    if piece is None:
        return

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
    else:
        # try promotion
        move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        if move in board.legal_moves:
            board.push(move)


# MAIN LOOP
if __name__ == "__main__":
    board = chess.Board("8/2ppk3/8/8/8/8/PPP1K3/8 w - - 0 1")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                handle_move(board)

        render(board)