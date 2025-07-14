from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal, Qt

class StartMenu(QWidget):
    start_game = pyqtSignal(str, int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Start Menu")
        self.setFixedSize(800, 600)

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                font-family: 'Segoe UI', sans-serif;
                font-size: 20px;
            }

            QLabel {
                background-color: transparent;
                color: black;
                font-weight: bold;
                margin-bottom: 10px;
                qproperty-alignment: AlignCenter;
            }

            QPushButton {
                background-color: #4caf50;
                border: none;
                border-radius: 15px;
                padding: 15px;
                font-size: 22px;
                font-weight: bold;
                color: white;
                margin-top: 30px;
            }

            QPushButton:hover {
                background-color: #66bb6a;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(150, 50, 150, 50)
        layout.setAlignment(Qt.AlignCenter)

        self.color_label = QLabel("Choose your color:")
        self.color_combo = QComboBox()
        self.color_combo.addItems(["White", "Black"])
        self.color_combo.setStyleSheet(self.get_combo_style("White"))

       
        self.elo_label = QLabel("Choose bot's ELO:")
        self.elo_combo = QComboBox()
        self.elo_combo.addItems(["800", "1200", "1600", "2000"])
        self.elo_combo.setStyleSheet(self.get_combo_style("800"))

        
        self.mode_label = QLabel("Choose mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["vs_bot", "2_players"])
        self.mode_combo.setStyleSheet(self.get_combo_style("vs_bot"))

        self.start_btn = QPushButton("Start ▶")
        self.start_btn.clicked.connect(self.handle_start)

        
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.color_label)
        layout.addWidget(self.color_combo)
        layout.addWidget(self.elo_label)
        layout.addWidget(self.elo_combo)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.start_btn)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.setLayout(layout)

        self.mode_combo.currentTextChanged.connect(self.toggle_elo_visibility)
        self.toggle_elo_visibility(self.mode_combo.currentText())

    def get_combo_style(self, highlight_text):
        
        return f"""
            QComboBox {{
                background-color: white;
                color: #4caf50;
                border: 2px solid #4caf50;
                border-radius: 10px;
                padding: 10px;
                font-size: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #1f1f1f;
                color: white;
                selection-background-color: #4caf50;
            }}
        """

    def toggle_elo_visibility(self, mode):
        if mode == "2_players":
            self.elo_label.hide()
            self.elo_combo.hide()
        else:
            self.elo_label.show()
            self.elo_combo.show()

    def handle_start(self):
        color = self.color_combo.currentText()
        elo = int(self.elo_combo.currentText())
        mode = self.mode_combo.currentText()
        self.start_game.emit(color.lower(), elo, mode)
