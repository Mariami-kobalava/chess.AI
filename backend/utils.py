from collections import deque
import re

import chess
import chess.pgn
import numpy as np
import torch

EVAL_PATTERN = re.compile(r"\[%eval\s+([-+]?\d+\.?\d*)]")

def filter_evaluated_games(input_path: str, output_path: str, game_limit = None):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by games; preserve '[Event' which we lose during split
    parts = content.strip().split('\n\n[Event ')
    games = [parts[0]] + [f'[Event {rest}' for rest in parts[1:]]

    # for g in games:
    #     print(g)
    valid_games = []

    for game in games:
        sections = game.strip().split('\n\n', 1)  # headers and moves

        if len(sections) != 2:
            continue  # malformed game

        headers, moves = sections

        # Check if every move line has an evaluation
        if all(EVAL_PATTERN.search(line) for line in moves.splitlines()):
            valid_games.append(game.strip())

        if game_limit is not None and len(valid_games) >= game_limit:
            break

    print(len(valid_games))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(valid_games))
        f.write('\n')  # end with newline


def board_to_tensor(history: list[chess.Board]):
    """
    Converts a list of up to 5 boards (current + last 4) into a tensor:
    shape (26,8,8) channel-first: 12 piece planes + side plane + 4 castling + 1 half-move + 8 history occupancy.
    history[0] = current board, history[1:] = last positions.
    """
    planes = []

    # piece planes for current position
    b = history[-1]

    for color in (chess.WHITE, chess.BLACK):
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            plane = np.zeros((8,8), dtype=np.float32)
            for sq in b.pieces(piece_type, color):
                r, c = divmod(sq, 8); plane[r, c] = 1
            planes.append(plane)

    # side-to-move
    stm = np.full((8,8), float(b.turn), dtype=np.float32)
    planes.append(stm)

    # castling rights
    for cr in (b.has_kingside_castling_rights(chess.WHITE),
               b.has_queenside_castling_rights(chess.WHITE),
               b.has_kingside_castling_rights(chess.BLACK),
               b.has_queenside_castling_rights(chess.BLACK)):
        planes.append(np.full((8,8), float(cr), dtype=np.float32))

    # halfmove clock normalized
    hm = min(b.halfmove_clock, 100) / 100.0
    planes.append(np.full((8,8), hm, dtype=np.float32))

    # history occupancy planes: 2 planes each for last 4 positions
    for h in history[:-1]:
        wplane = np.zeros((8,8), dtype=np.float32)
        bplane = np.zeros((8,8), dtype=np.float32)

        if h is None:
            planes.extend([wplane, bplane])
            continue

        # bb = h.occupied_co[color]
        # bits = ((bb >> np.arange(64)) & 1).reshape(8, 8).astype(np.float32)
        for sq in chess.SquareSet(h.occupied_co[chess.WHITE]):
            r, c = divmod(sq, 8)
            wplane[r, c] = 1

        for sq in chess.SquareSet(h.occupied_co[chess.BLACK]):
            r, c = divmod(sq, 8)
            bplane[r, c] = 1

        planes.extend([wplane, bplane])

    # stack and convert to tensor
    data = np.stack(planes, axis=0)

    return torch.from_numpy(data)

# board=chess.Board()
# print(board_to_tensor([board]).size())

def create_board_history(board: chess.Board):
    board_history = deque(maxlen=5)  # 4 past + current
    history_length = 4

    move_stack = board.move_stack
    num_moves = len(move_stack)

    # Determine how many placeholders we need
    missing = history_length - min(history_length, num_moves)

    # Add None placeholders for missing history
    for _ in range(missing):
        board_history.append(None)

    # Add actual past board states
    for i in range(-min(history_length, num_moves), 0):
        tmp = chess.Board()
        for m in move_stack[:i + len(move_stack)]:
            tmp.push(m)
        board_history.append(tmp)

    board_history.append(board.copy())

    return board_history

def iter_games(pgn_paths):
    """
    Yield each game from PGN files, one at a time.
    PGN structure: tags, blank line, moves, blank line, repeat.
    """
    for path in pgn_paths:
        with open(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game