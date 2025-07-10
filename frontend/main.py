import sys
import os

import chess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.cnn_inference import init_model

# აქ შეგიძლია გამოიყენო init_model ფუნქცია, მაგალითად:


# from backend.cnn_inference import init_model
from backend.mcts2 import ChessAI

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication, QStackedWidget
from ui.start_menu import StartMenu
from ui.chess_board import ChessBoard

class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Chess")
        self.setFixedSize(600, 650)

        self.start_menu = StartMenu()
        self.start_menu.start_game.connect(self.launch_game)

        self.addWidget(self.start_menu)
        self.setCurrentWidget(self.start_menu)
    
    def launch_game(self, color, elo, mode):
        if color == "White":    
            color = chess.WHITE
        else: 
            color == chess.BLACK

        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chess_model.pth'))
        model, device = init_model(model_path)

        mcts = ChessAI(model, color, device=device)
        self.chess_board = ChessBoard(color, elo, mode, mcts)
        self.addWidget(self.chess_board)
        self.setCurrentWidget(self.chess_board)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

