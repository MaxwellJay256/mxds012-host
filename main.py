"""
MXDS012 Host - ADC data acquisition and visualization application.

A thin entry point that creates and runs the main application window.
All business logic is delegated to modules: serial_reader, protocol, plot_panel, recording_*.
"""
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from gui.MainWindow import MainWindow


def main():
    """Application entry point."""
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

