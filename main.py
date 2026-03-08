import sys
from PyQt6.QtWidgets import QApplication

# Now Python can see the 'ui' folder because we are running from the root!
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())