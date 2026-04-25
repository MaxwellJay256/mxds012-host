"""Right-side function area with pivot navigation and stacked pages."""
from PyQt5.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import Pivot

from .RecordingWidget import RecordingWidget
from .PlaybackWidget import PlaybackWidget
from .RealtimeAnalyzeWidget import RealtimeAnalyzeWidget
from utils.PreprocessParamService import PreprocessParamService


class FunctionArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recordingWidget = RecordingWidget(self)
        self.playbackWidget = PlaybackWidget(self)
        self.realtimeAnalyzeWidget = RealtimeAnalyzeWidget(self)

        self.pivot = Pivot(self)
        self.stackedWidget = QStackedWidget(self)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)

        pivot_row = QHBoxLayout()
        # pivot_row.addStretch()
        pivot_row.addWidget(self.pivot)
        pivot_row.addStretch()
        root_layout.addLayout(pivot_row)

        root_layout.addWidget(self.stackedWidget, 1)

        self.addPage(self.recordingWidget, "recording", "录制")
        self.addPage(self.playbackWidget, "playback", "回放")
        self.addPage(self.realtimeAnalyzeWidget, "realtimeAnalyze", "在线识别")

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentIndex(0)
        self.pivot.setCurrentItem(self.recordingWidget.objectName())

    def addPage(self, widget: QWidget, objectName: str, text: str):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)

        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda *_, widget=widget: self.stackedWidget.setCurrentWidget(widget),
        )

    def onCurrentIndexChanged(self, index: int):
        widget = self.stackedWidget.widget(index)
        if widget is None:
            return
        self.pivot.setCurrentItem(widget.objectName())

        if widget is self.realtimeAnalyzeWidget:
            main_window = self.window()
            plot_panel = getattr(main_window, "plotPanel", None)
            if plot_panel is not None:
                plot_panel.setRealtimePreprocessEnabled(True, emit_signal=True)

            preprocess_service = PreprocessParamService.get_instance()
            preprocess_service.update_param("standardizePerFrame", True)
