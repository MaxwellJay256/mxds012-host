"""实时超声预处理工具。"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class PreprocessParams:
	"""实时超声预处理参数。

	属性说明:
		sampling_freq_hz: ADC 采样频率，单位 Hz
		tgc_alpha: 
		sampling_balance: TGC 前用于去直流的 ADC 基线值
		burst_tone_freq_hz: 用于 TGC 缩放的发射中心频率
		inspect_depth_m: 构造深度增益曲线时使用的等效探测深度（米）
		butter_low_hz_list: 4 个通道各自的巴特沃斯带通低截止频率，单位 Hz
		butter_high_hz_list: 4 个通道各自的巴特沃斯带通高截止频率，单位 Hz
		butter_order_list: 4 个通道各自的巴特沃斯滤波器阶数
		enable_log_compression: 是否启用对数压缩
		log_comp_ratio: 启用对数压缩时的压缩系数
		standardize_per_frame: 是否对每一帧（4通道整体）做标准化
	"""

	samplingFreqHz: float = 5.5e6  # ADC 采样频率，Hz
	tgcAlpha: float = 0.05  # 时间增益补偿（TGC）强度系数，越大表示增益随深度增加得越快
	samplingBalance: float = 0  # TGC 前用于去直流的 ADC 基线值
	burstToneFreqHz: float = 4e6  # 用于 TGC 缩放的发射中心频率，Hz
	inspectDepth_m: float = 70.0e-3  # 构造深度增益曲线时使用的等效探测深度，单位米
	butterLowHzList: List[float] = field(default_factory=lambda: [4e6, 3e6, 3e6, 3e6])  # 巴特沃斯带通低截止频率，Hz
	butterHighHzList: List[float] = field(default_factory=lambda: [6e6, 7e6, 7e6, 7e6])  # 巴特沃斯带通高截止频率，Hz
	butterOrderList: List[int] = field(default_factory=lambda: [5, 4, 4, 4])  # 巴特沃斯滤波器阶数
	enableLogCompression: bool = False  # 是否启用对数压缩
	logCompRatio: float = 0.1  # 对数压缩系数，越大表示压缩越激烈
	standardizePerFrame: bool = False  # 是否启用逐帧标准化
	normalizePerFrame: bool = False  # 是否启用逐帧归一化


class RealtimePreprocessor:
	"""面向 4 通道超声 ADC 数据的逐帧实时预处理器。"""

	def __init__(self, params: Optional[PreprocessParams] = None):
		self.params = params or PreprocessParams()
		self._channel_filters = self._buildChannelFilters()
		self._b, self._a = self._channel_filters[1]

	def _buildButterBandpass(self, low_hz: float, high_hz: float, order: int):
		nyquist = max(1.0, self.params.samplingFreqHz * 0.5)
		low_hz = max(1.0, float(low_hz))
		high_hz = max(low_hz + 1.0, float(high_hz))

		# 对高截止频率进行钳位，确保归一化频率位于 (0, 1)。
		if high_hz >= nyquist:
			high_hz = nyquist * 0.95
		if low_hz >= high_hz:
			low_hz = high_hz * 0.5

		low = low_hz / nyquist
		high = high_hz / nyquist
		butter_result = signal.butter(int(order), [low, high], btype="band", output="ba")
		if butter_result is None or len(butter_result) < 2:
			raise ValueError("Failed to build Butterworth coefficients")
		b = np.asarray(butter_result[0], dtype=np.float64)
		a = np.asarray(butter_result[1], dtype=np.float64)
		return b, a

	def _buildChannelFilters(self):
		"""为每个通道构建独立的带通滤波器；通道1更激进，其余通道使用默认参数。"""
		low_list = list(self.params.butterLowHzList)
		high_list = list(self.params.butterHighHzList)
		order_list = list(self.params.butterOrderList)

		if len(low_list) != 4 or len(high_list) != 4 or len(order_list) != 4:
			raise ValueError("butterLowHzList/butterHighHzList/butterOrderList 必须都为 4 个元素")

		channel_filters = []
		for low_hz, high_hz, order in zip(low_list, high_list, order_list):
			channel_filters.append(self._buildButterBandpass(low_hz, high_hz, order))
		return channel_filters

	def _buildTGC(self, depth: int) -> np.ndarray:
		depth = max(1, int(depth))
		m_depth = np.arange(depth, dtype=np.float32) * (self.params.inspectDepth_m / float(depth))  # 
		tgc = np.exp(self.params.tgcAlpha * self.params.burstToneFreqHz / 1e6 * m_depth * 100.0)
		return tgc.astype(np.float32, copy=False)

	def _filtfiltOrLfilter(self, values: np.ndarray) -> np.ndarray:
		'''
		对于长度较短的信号，filtfilt 可能无法正常工作（会抛出 ValueError）。此时退回到 lfilter。
		'''
		return self._filtfiltOrLfilterWithCoefficients(values, self._b, self._a)

	def _filtfiltOrLfilterWithCoefficients(
		self,
		values: np.ndarray,
		b: np.ndarray,
		a: np.ndarray,
	) -> np.ndarray:
		"""使用指定滤波器系数处理信号；短信号时回退为 lfilter。"""
		if values.size < 3:
			return values

		padlen_default = 3 * (max(len(a), len(b)) - 1)
		if values.size <= padlen_default:
			filtered_short = signal.lfilter(b, a, values)
			return np.asarray(filtered_short, dtype=np.float32)

		filtered = signal.filtfilt(b, a, values)
		return np.asarray(filtered, dtype=np.float32)

	def _toIntFrame(self, processed: np.ndarray) -> List[List[int]]:
		processed_int = np.rint(np.abs(processed)).astype(np.int32, copy=False)
		return [processed_int[idx, :].tolist() for idx in range(4)]

	def _toFloatFrame(self, processed: np.ndarray) -> List[List[float]]:
		processed_float = np.asarray(processed, dtype=np.float32)
		return [processed_float[idx, :].tolist() for idx in range(4)]

	def _buildEnvelopeFrame(self, channels: Iterable[Iterable[float]]) -> np.ndarray:
		"""执行到包络/可选对数压缩步骤，返回 float32 帧。"""
		channel_arrays = [np.asarray(channel, dtype=np.float32).reshape(-1) for channel in channels]
		if len(channel_arrays) != 4:
			raise ValueError(f"Expected 4 channels, got {len(channel_arrays)}")

		if not channel_arrays or channel_arrays[0].size == 0:
			return np.zeros((4, 0), dtype=np.float32)

		depths = [arr.size for arr in channel_arrays]
		if len(set(depths)) != 1:
			raise ValueError(f"Channel depth mismatch: {depths}")

		depth = depths[0]  # 所有通道深度相同，取第一个即可

		tgc = self._buildTGC(depth)

		processed = np.zeros((4, depth), dtype=np.float32)
		for ch_no, input_channel in enumerate(channel_arrays):
			b, a = self._channel_filters[ch_no]
			# 1. 去直流并应用 TGC
			temp = (input_channel - self.params.samplingBalance) * tgc
			# 2. 巴特沃斯带通滤波
			temp = self._filtfiltOrLfilterWithCoefficients(temp, b, a)
			# 3. 包络检测
			analytic = np.asarray(signal.hilbert(temp), dtype=np.complex64)
			envelope = np.abs(analytic).astype(np.float32, copy=False)

			# 4. 对数压缩
			if self.params.enableLogCompression:
				ratio = max(1e-9, float(self.params.logCompRatio))
				envelope = np.log10(1.0 + ratio * envelope) / np.log10(1.0 + ratio)

			processed[ch_no, :] = envelope

		return processed

	def _applyPostProcess(self, processed: np.ndarray) -> np.ndarray:
		"""应用标准化与归一化等后处理步骤。"""
		if self.params.standardizePerFrame:
			mean = float(np.mean(processed))
			std = float(np.std(processed)) + 1e-6
			processed = (processed - mean) / std

		if self.params.normalizePerFrame:
			min_val = float(np.min(processed))
			max_val = float(np.max(processed))
			if max_val - min_val > 1e-6:
				processed = (processed - min_val) / (max_val - min_val) * 512.0
			else:
				processed = np.zeros_like(processed)
		return processed

	def processForOutputAndDisplay(self, channels: Iterable[Iterable[float]]) -> Tuple[List[List[int]], List[List[int]]]:
		"""返回 (输出分支, 显示分支) 两路数据。"""
		envelope = self._buildEnvelopeFrame(channels)
		output_processed = self._applyPostProcess(envelope.copy())

		output_frame = self._toIntFrame(output_processed)
		if self.params.standardizePerFrame:
			display_frame = self._toIntFrame(envelope)
		else:
			display_frame = output_frame
		return output_frame, display_frame

	def processForModelOutputAndDisplay(
		self,
		channels: Iterable[Iterable[float]],
	) -> Tuple[List[List[float]], List[List[int]], List[List[int]]]:
		"""返回 (模型分支, 输出分支, 显示分支) 三路数据。"""
		envelope = self._buildEnvelopeFrame(channels)
		model_processed = self._applyPostProcess(envelope.copy())

		model_frame = self._toFloatFrame(model_processed)
		output_frame = self._toIntFrame(model_processed)
		if self.params.standardizePerFrame:
			display_frame = self._toIntFrame(envelope)
		else:
			display_frame = output_frame
		return model_frame, output_frame, display_frame

	def process(self, channels: Iterable[Iterable[float]]) -> List[List[int]]:
		"""处理一帧数据，输入输出形状均为 (通道, 深度)。"""
		output_frame, _ = self.processForOutputAndDisplay(channels)
		return output_frame


def Preprocessing(data, mode: str = "standardize_only"):
	"""Compatibility helper used by the flat dataset loader.

	The input is expected to have shape (4, depth, frames) and the return value
	keeps the same shape, but uses float32 values.
	"""
	array = np.asarray(data, dtype=np.float32)
	if array.ndim != 3:
		raise ValueError(f"Expected 3D data array, got shape={array.shape}")

	if mode == "raw":
		return array.astype(np.float32, copy=False)

	params = PreprocessParams()
	params.standardizePerFrame = True
	params.normalizePerFrame = mode in {"standardize_and_normalize", "normalize", "full"}
	preprocessor = RealtimePreprocessor(params)

	frames = []
	for frame_idx in range(array.shape[2]):
		frame = array[:, :, frame_idx]
		model_frame, _, _ = preprocessor.processForModelOutputAndDisplay(frame)
		frames.append(np.asarray(model_frame, dtype=np.float32))

	if not frames:
		return np.zeros_like(array, dtype=np.float32)

	return np.stack(frames, axis=2).astype(np.float32, copy=False)

