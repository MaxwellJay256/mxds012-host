"""Training page scaffold for dataset analysis and model training."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPlainTextEdit
from qfluentwidgets import BodyLabel, DoubleSpinBox, FluentIcon, HeaderCardWidget, LineEdit, PrimaryPushButton, PushButton, SingleDirectionScrollArea, SpinBox


class ClickableLineEdit(LineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, a0):
        self.clicked.emit()
        super().mousePressEvent(a0)


class TrainingWidget(QWidget):
    startTrainingRequested = pyqtSignal(dict)
    refreshRequested = pyqtSignal(str)
    infoRequested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._dataset_root = ""

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        scroll_area = SingleDirectionScrollArea(self)
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        dataset_card = HeaderCardWidget(content)
        dataset_card.setTitle("数据集分析")
        dataset_layout = QVBoxLayout()
        dataset_layout.setSpacing(8)

        dataset_row = QHBoxLayout()
        dataset_row.setSpacing(10)
        self.datasetPathEdit = ClickableLineEdit(self)
        self.datasetPathEdit.setReadOnly(True)
        self.datasetPathEdit.setPlaceholderText("选择训练数据集文件夹")
        self.datasetBrowseButton = PushButton(FluentIcon.FOLDER, "选择文件夹")
        self.datasetRefreshButton = PushButton(FluentIcon.SYNC, "分析")
        dataset_row.addWidget(self.datasetPathEdit, 1)
        dataset_row.addWidget(self.datasetBrowseButton)
        dataset_row.addWidget(self.datasetRefreshButton)
        dataset_layout.addLayout(dataset_row)

        self.datasetSummaryLabel = BodyLabel("尚未选择数据集")
        dataset_layout.addWidget(self.datasetSummaryLabel)

        self.datasetIssueLabel = BodyLabel("")
        self.datasetIssueLabel.setWordWrap(True)
        dataset_layout.addWidget(self.datasetIssueLabel)

        dataset_card.viewLayout.addLayout(dataset_layout)
        content_layout.addWidget(dataset_card)

        training_card = HeaderCardWidget(content)
        training_card.setTitle("训练参数")
        training_layout = QVBoxLayout()
        training_layout.setSpacing(8)

        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        self.modelNameEdit = LineEdit(self)
        self.modelNameEdit.setPlaceholderText("模型名称")
        self.epochsSpin = SpinBox(self)
        self.epochsSpin.setRange(1, 1000)
        self.epochsSpin.setValue(50)
        self.batchSizeSpin = SpinBox(self)
        self.batchSizeSpin.setRange(1, 1024)
        self.batchSizeSpin.setValue(64)
        self.batchSizeSpin.setFixedWidth(120)
        self.lrEdit = LineEdit(self)
        self.lrEdit.setText("0.001")
        self.lrEdit.setPlaceholderText("学习率，例如 0.001")
        model_row.addWidget(self.modelNameEdit, 1)
        model_row.addWidget(self.epochsSpin)
        model_row.addWidget(self.batchSizeSpin)
        model_row.addWidget(self.lrEdit)
        training_layout.addLayout(model_row)

        split_row = QHBoxLayout()
        split_row.setSpacing(10)

        self.trainRatioSpin = DoubleSpinBox(self)
        self.trainRatioSpin.setRange(0.0, 1.0)
        self.trainRatioSpin.setSingleStep(0.05)
        self.trainRatioSpin.setDecimals(2)
        self.trainRatioSpin.setValue(0.8)
        self.trainRatioSpin.setPrefix("训练 ")

        self.valRatioSpin = DoubleSpinBox(self)
        self.valRatioSpin.setRange(0.0, 1.0)
        self.valRatioSpin.setSingleStep(0.05)
        self.valRatioSpin.setDecimals(2)
        self.valRatioSpin.setValue(0.1)
        self.valRatioSpin.setPrefix("验证 ")

        self.testRatioSpin = DoubleSpinBox(self)
        self.testRatioSpin.setRange(0.0, 1.0)
        self.testRatioSpin.setSingleStep(0.05)
        self.testRatioSpin.setDecimals(2)
        self.testRatioSpin.setValue(0.1)
        self.testRatioSpin.setPrefix("测试 ")

        split_row.addWidget(self.trainRatioSpin)
        split_row.addWidget(self.valRatioSpin)
        split_row.addWidget(self.testRatioSpin)
        split_row.addStretch()
        training_layout.addLayout(split_row)

        output_row = QHBoxLayout()
        output_row.setSpacing(10)
        self.outputPathEdit = ClickableLineEdit(self)
        self.outputPathEdit.setReadOnly(True)
        self.outputPathEdit.setPlaceholderText("选择模型输出路径")
        self.outputBrowseButton = PushButton(FluentIcon.FOLDER, "选择输出")
        self.startTrainingButton = PrimaryPushButton(FluentIcon.PLAY_SOLID, "开始训练")
        output_row.addWidget(self.outputPathEdit, 1)
        output_row.addWidget(self.outputBrowseButton)
        output_row.addWidget(self.startTrainingButton)
        training_layout.addLayout(output_row)

        self.logView = QPlainTextEdit(self)
        self.logView.setReadOnly(True)
        self.logView.setPlaceholderText("训练日志将显示在这里")
        self.logView.setMinimumHeight(220)
        training_layout.addWidget(self.logView)

        training_card.viewLayout.addLayout(training_layout)
        content_layout.addWidget(training_card)
        content_layout.addStretch(1)

        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area, 1)

        self.datasetBrowseButton.clicked.connect(self._chooseDatasetDir)
        self.datasetPathEdit.clicked.connect(self._chooseDatasetDir)
        self.datasetRefreshButton.clicked.connect(self._emitRefresh)
        self.outputBrowseButton.clicked.connect(self._chooseOutputPath)
        self.outputPathEdit.clicked.connect(self._chooseOutputPath)
        self.startTrainingButton.clicked.connect(self._emitStartTraining)

    def _chooseDatasetDir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择训练数据集文件夹", self._dataset_root or str(Path.cwd()))
        if not directory:
            return
        self._dataset_root = directory
        self.datasetPathEdit.setText(directory)
        self.refreshRequested.emit(directory)

    def _chooseOutputPath(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择模型输出路径", "cbam_lstm_best.onnx", "ONNX Model (*.onnx)")
        if not file_path:
            return
        if not file_path.lower().endswith(".onnx"):
            file_path += ".onnx"
        self.outputPathEdit.setText(file_path)

    def _emitRefresh(self):
        dataset_dir = self.datasetPathEdit.text().strip()
        if not dataset_dir:
            self.infoRequested.emit("请先选择数据集文件夹", "warning")
            return
        self.refreshRequested.emit(dataset_dir)

    def _emitStartTraining(self):
        dataset_dir = self.datasetPathEdit.text().strip()
        output_path = self.outputPathEdit.text().strip()
        model_name = self.modelNameEdit.text().strip()
        lr_text = self.lrEdit.text().strip()

        if not dataset_dir:
            self.infoRequested.emit("请先选择数据集文件夹", "warning")
            return
        if not output_path:
            self.infoRequested.emit("请先选择模型输出路径", "warning")
            return
        if not model_name:
            self.infoRequested.emit("请输入模型名称", "warning")
            return

        try:
            learning_rate = float(lr_text)
        except ValueError:
            self.infoRequested.emit("请输入有效的学习率", "warning")
            return

        payload = {
            "dataset_dir": dataset_dir,
            "model_name": model_name,
            "epochs": int(self.epochsSpin.value()),
            "batch_size": int(self.batchSizeSpin.value()),
            "lr": learning_rate,
            "train_ratio": float(self.trainRatioSpin.value()),
            "val_ratio": float(self.valRatioSpin.value()),
            "test_ratio": float(self.testRatioSpin.value()),
            "output_path": output_path,
        }
        self.startTrainingRequested.emit(payload)

    def appendLog(self, message: str):
        self.logView.appendPlainText(str(message))

    def setTrainingRunning(self, running: bool):
        self.startTrainingButton.setEnabled(not bool(running))

    def setDatasetSummary(self, summary: str):
        self.datasetSummaryLabel.setText(summary)

    def setDatasetIssues(self, issues: list[str]):
        if not issues:
            self.datasetIssueLabel.setText("未发现明显问题")
            return
        preview = issues[:5]
        suffix = "" if len(issues) <= 5 else f" ... 共 {len(issues)} 项"
        self.datasetIssueLabel.setText("\n".join(preview) + suffix)
