# STM32G474 ADC 上位机

基于 PyQt5 与 PyQt-Fluent-Widgets 的示例上位机，可通过 USB 虚拟串口接收 STM32G474 提供的 5 路 ADC 帧数据，对前四路减去参考通道后实时绘图。

## 运行环境

- Python 3.12.12
- PyQt5
- PyQt-Fluent-Widgets (`pip install PyQt-Fluent-Widgets`)
- pyqtgraph
- pyserial

建议在已准备好的 Conda 虚拟环境中安装依赖：

```powershell
pip install PyQt5 PyQt-Fluent-Widgets pyqtgraph pyserial
```

## 使用说明

1. 运行 `python main.py` 启动界面。
2. 在“刷新串口”后选择对应的 STM32 虚拟串口。
3. 点击“开始接收”后将每 50 ms 刷新 4 张曲线图；曲线对应 `通道1~4 - 通道5`。
4. 点击“停止”可关闭串口。

## 数据帧格式建议

- 建议确保每个通道的采样点数一致，并避免额外的空白字符。
- 如有可能，可在帧首增加采样点数或时间戳，便于上位机校验。
