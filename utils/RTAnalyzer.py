"""Realtime ONNX analyzer for ultrasound gesture recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
from collections import Counter, deque

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

_ORT_MODULE = None
_ORT_IMPORT_ERROR = None


def _get_onnxruntime():
	global _ORT_MODULE, _ORT_IMPORT_ERROR
	if _ORT_MODULE is not None:
		return _ORT_MODULE
	if _ORT_IMPORT_ERROR is not None:
		raise RuntimeError(f"未安装 onnxruntime，请先安装后再使用在线识别: {_ORT_IMPORT_ERROR}")

	try:
		import onnxruntime as ort  # type: ignore[import-not-found]
		_ORT_MODULE = ort
		return _ORT_MODULE
	except Exception as error:  # pragma: no cover - import availability depends on runtime env
		_ORT_IMPORT_ERROR = error
		raise RuntimeError(f"未安装 onnxruntime，请先安装后再使用在线识别: {error}") from error


class RTAnalyzer(QObject):
	"""Realtime analyzer that converts processed frames into model probabilities."""

	inferenceReady = pyqtSignal(dict)
	warningRaised = pyqtSignal(str)
	errorRaised = pyqtSignal(str)

	_instance: Optional["RTAnalyzer"] = None

	@classmethod
	def getInstance(cls) -> "RTAnalyzer":
		if cls._instance is None:
			cls._instance = cls()
		return cls._instance

	def __init__(self, parent=None):
		super().__init__(parent)
		self._enabled = False
		self._session = None
		self._input_name = ""
		self._classes: List[str] = []
		self._model_path = ""
		self._expected_depth = 0
		self._model_type = "cnn2d"
		self._ema_alpha = 0.5
		self._ema_probs: Optional[np.ndarray] = None
		# Majority vote buffer
		self._vote_window_size = 5
		self._vote_buffer: deque = deque(maxlen=self._vote_window_size)
		self._last_output_name: Optional[str] = None
		self._last_warn_output_mismatch = False

	def setEnabled(self, enabled: bool):
		self._enabled = bool(enabled)
		if not self._enabled:
			self.resetEma()

	def isEnabled(self) -> bool:
		return self._enabled

	def setEmaAlpha(self, alpha: float):
		try:
			value = float(alpha)
		except (TypeError, ValueError):
			value = 0.5
		self._ema_alpha = max(0.0, min(1.0, value))

	def setVoteWindowSize(self, n: int):
		try:
			value = int(n)
		except (TypeError, ValueError):
			value = 5
		value = max(1, min(99, value))
		self._vote_window_size = value
		old = self._vote_buffer
		self._vote_buffer = deque(list(old), maxlen=self._vote_window_size)

	def getVoteWindowSize(self) -> int:
		return int(self._vote_window_size)

	def resetEma(self):
		self._ema_probs = None
		# also reset voting state
		self._vote_buffer.clear()
		self._last_output_name = None

	def clearModel(self):
		self._session = None
		self._input_name = ""
		self._classes = []
		self._model_path = ""
		self._expected_depth = 0
		self._model_type = "cnn2d"
		self._ema_probs = None
		self._vote_buffer.clear()
		self._last_output_name = None
		self._last_warn_output_mismatch = False

	def hasModel(self) -> bool:
		return self._session is not None and bool(self._classes)

	def loadModel(self, model_path: str, classes: Iterable[str], expected_depth: int = 0, model_type: str = "cbamlstm"):
		model_path = str(model_path or "").strip()
		class_list = [str(name).strip() for name in classes if str(name).strip()]
		if not model_path:
			raise ValueError("模型路径为空")
		if not class_list:
			raise ValueError("模型类别为空")
		if Path(model_path).suffix.lower() != ".onnx":
			raise ValueError("仅支持 .onnx 模型")
		ort = _get_onnxruntime()

		try:
			session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
		except Exception as error:
			raise RuntimeError(f"创建 ONNX 推理会话失败: {error}") from error

		input_meta = session.get_inputs()
		if not input_meta:
			raise RuntimeError("模型输入为空")

		self._session = session
		self._input_name = str(input_meta[0].name)
		self._classes = class_list
		self._model_path = model_path
		self._expected_depth = max(0, int(expected_depth))
		normalized_type = str(model_type).strip().lower()
		if normalized_type not in {"cbamlstm", "cnn2d"}:
			normalized_type = "cbamlstm"
		self._model_type = normalized_type
		self._ema_probs = None
		self._last_warn_output_mismatch = False

	def analyzeFrame(self, channels: List[List[float]]):
		if not self._enabled or self._session is None:
			return

		try:
			tensor = self._buildInputTensor(channels)
			outputs = self._session.run(None, {self._input_name: tensor})
			probabilities = self._extractProbabilities(outputs)
			if probabilities.size == 0:
				return

			class_count = min(len(self._classes), int(probabilities.size))
			if class_count <= 0:
				return

			if class_count != len(self._classes) and not self._last_warn_output_mismatch:
				self._last_warn_output_mismatch = True
				self.warningRaised.emit(
					f"模型输出维度({int(probabilities.size)})与 classes 数量({len(self._classes)})不一致，已按最小长度对齐"
				)

			aligned_probs = probabilities[:class_count]
			aligned_classes = self._classes[:class_count]
			smoothed_probs = self._smoothProbabilities(aligned_probs)

			top_index = int(np.argmax(smoothed_probs))
			top_name = aligned_classes[top_index]
			top_prob = float(smoothed_probs[top_index])

			probs_by_name: Dict[str, float] = {
				name: float(prob)
				for name, prob in zip(aligned_classes, smoothed_probs)
			}

			# Append current top to vote buffer and compute majority
			try:
				self._vote_buffer.append(top_name)
			except Exception:
				# ensure buffer exists
				self._vote_buffer = deque([top_name], maxlen=self._vote_window_size)

			counts = Counter(self._vote_buffer)
			most_common_name, most_common_count = counts.most_common(1)[0]
			threshold = (int(self._vote_window_size) + 1) // 2

			if most_common_count >= threshold:
				final_top_name = most_common_name
			else:
				# keep previous output if exists, otherwise use current top
				if self._last_output_name is not None:
					final_top_name = self._last_output_name
				else:
					final_top_name = top_name

			self._last_output_name = final_top_name
			final_top_prob = float(probs_by_name.get(final_top_name, top_prob))
			highlight = final_top_prob >= 0.5

			self.inferenceReady.emit(
				{
					"model_path": self._model_path,
					"classes": list(aligned_classes),
					"probabilities": probs_by_name,
					"top_name": final_top_name,
					"top_prob": final_top_prob,
					"highlight": bool(highlight),
				}
			)
		except Exception as error:
			self.errorRaised.emit(f"在线识别失败: {error}")

	def _smoothProbabilities(self, raw_probs: np.ndarray) -> np.ndarray:
		"""
		EMA 平滑概率值，适用于在线识别场景，能在一定程度上减少短时波动带来的误识别。
		EMA 公式: ema_t = α * raw_t + (1 - α) * ema_{t-1}
		"""
		raw_probs = np.asarray(raw_probs, dtype=np.float32).reshape(-1)
		if raw_probs.size == 0:
			return raw_probs

		if self._ema_probs is None or self._ema_probs.size != raw_probs.size:
			self._ema_probs = raw_probs.astype(np.float32, copy=True)
			return self._ema_probs.astype(np.float32, copy=True)

		self._ema_probs = (
			self._ema_alpha * raw_probs
			+ (1.0 - self._ema_alpha) * self._ema_probs
		).astype(np.float32, copy=False)
		return self._ema_probs.astype(np.float32, copy=True)

	def _buildInputTensor(self, channels: List[List[float]]) -> np.ndarray:
		if not isinstance(channels, list) or len(channels) != 4:
			raise ValueError("输入帧必须是 4 通道")

		arrays = [np.asarray(channel, dtype=np.float32).reshape(-1) for channel in channels]
		lengths = [int(arr.size) for arr in arrays]
		if not lengths or min(lengths) <= 0:
			raise ValueError("输入帧为空")
		if len(set(lengths)) != 1:
			raise ValueError(f"通道深度不一致: {lengths}")

		depth = lengths[0]
		if self._expected_depth > 0 and depth != self._expected_depth:
			self.warningRaised.emit(
				f"当前采样深度 {depth} 与模型期望 {self._expected_depth} 不一致，已按当前输入推理"
			)

		frame = np.stack(arrays, axis=0)
		if self._model_type == "cnn2d":
			# CNN2D 约定输入: (batch, c, h, w)
			return frame.reshape(1, 1, 4, depth).astype(np.float32, copy=False)
		# CBAMLSTM 约定输入: (batch, t_seq, c, h, w)
		return frame.reshape(1, 1, 1, 4, depth).astype(np.float32, copy=False)

	def _extractProbabilities(self, outputs) -> np.ndarray:
		if not outputs:
			raise ValueError("模型输出为空")

		logits = np.asarray(outputs[0], dtype=np.float32)
		logits = logits.reshape(-1)
		if logits.size == 0:
			return np.asarray([], dtype=np.float32)

		logits = logits - np.max(logits)
		exp_logits = np.exp(logits)
		denom = float(np.sum(exp_logits))
		if denom <= 1e-12:
			raise ValueError("softmax 分母为 0")
		probs = exp_logits / denom
		return probs.astype(np.float32, copy=False)
