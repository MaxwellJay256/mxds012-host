"""
MXDS012 Host - ADC data acquisition and visualization application.

A thin entry point that creates and runs the main application window.
All business logic is delegated to modules: serial_reader, protocol, plot_panel, recording_*.
"""
import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow



def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

