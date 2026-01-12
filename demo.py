'''
本程序用来演示 QSettings 的基本用法。
它会保存一个文本框的内容和一个复选框的状态，并在程序重启后恢复这些设置。

QSettings 会将设置存储在平台相关的位置，例如 Windows 注册表、macOS 的 plist 文件或 Linux 的配置文件中。
'''
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QCheckBox, QVBoxLayout
from PyQt5.QtCore import QSettings

class Demo(QWidget):
    def __init__(self):
        super().__init__()
        self.edit = QLineEdit()
        self.chk = QCheckBox("启用高级选项")
        lay = QVBoxLayout(self)
        lay.addWidget(self.edit)
        lay.addWidget(self.chk)

        # 初始化 settings（建议用组织名/应用名）
        self.settings = QSettings("KeineStudio", "DemoApp")
        self.load_settings()

    def load_settings(self):
        # 读取时提供默认值
        self.edit.setText(self.settings.value("edit_text", "", type=str))
        self.chk.setChecked(self.settings.value("advanced_enabled", False, type=bool))

    def closeEvent(self, event):
        # 关闭前保存
        self.settings.setValue("edit_text", self.edit.text())
        self.settings.setValue("advanced_enabled", self.chk.isChecked())
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = Demo()
    w.show()
    sys.exit(app.exec_())
