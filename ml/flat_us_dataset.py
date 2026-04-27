import os
import re
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency during app startup
    torch = None

    TorchDataset = object  # type: ignore[assignment]

    _TORCH_AVAILABLE = False

torch = cast(Any, torch)

from utils.PreProcess import Preprocessing
from utils.GestureConstants import DEFAULT_GESTURE_ORDER

_FILENAME_RE = re.compile(r'^(?P<subject>[^-]+)-(?P<gesture>[A-Za-z]+)-(?P<trial>\d+)\.txt$')
_CHANNEL_RE = re.compile(r'^CH(?P<index>\d+)$', re.IGNORECASE)


class DataFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileMeta:
    path: str
    subject: str
    gesture: str
    trial: int


@dataclass
class ParseIssue:
    path: str
    frame_index: int
    line_number: int
    message: str


@dataclass
class ParsedRecord:
    meta: FileMeta
    frames: np.ndarray  # (num_frames, 4, depth)


@dataclass
class SplitConfig:
    strategy: str = 'ratio'  # ratio - 按比例分割 | trial - 按试验分割 | explicit - 显式指定
    seed: int = 42  # 随机分割种子
    train_ratio: float = 0.8  # 训练集比例
    val_ratio: float = 0.1 # 验证集比例
    test_ratio: float = 0.1 # 测试集比例
    train_trials: Tuple[int, ...] = (1,)
    val_trials: Tuple[int, ...] = (2,)
    test_trials: Tuple[int, ...] = tuple()
    explicit_map: Optional[Dict[str, str]] = None


class FlatUltrasoundParser:
    """Parser for flat files with line format: Fxxxxxx CHx s1 s2 ... sN."""

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.issues: List[ParseIssue] = []
        self.ignored_files: List[str] = []
        self.global_depth: Optional[int] = None

    def discover(self, root_dir: str) -> List[FileMeta]:
        metas: List[FileMeta] = []
        for name in sorted(os.listdir(root_dir)):
            full_path = os.path.join(root_dir, name)
            if not os.path.isfile(full_path):
                continue
            match = _FILENAME_RE.match(name)
            if not match:
                self.ignored_files.append(full_path)
                continue
            metas.append(
                FileMeta(
                    path=full_path,
                    subject=match.group('subject'),
                    gesture=match.group('gesture').upper(),
                    trial=int(match.group('trial')),
                )
            )
        return metas

    def parse_file(self, meta: FileMeta) -> Optional[ParsedRecord]:
        with open(meta.path, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        if len(raw_lines) == 0:
            return self._fail_or_skip(meta.path, -1, -1, 'Empty file')

        if len(raw_lines) % 4 != 0:
            return self._fail_or_skip(
                meta.path, -1, -1,
                f'Line count {len(raw_lines)} is not divisible by 4'
            )

        frame_list: List[np.ndarray] = []
        frame_count = len(raw_lines) // 4

        for frame_idx in range(frame_count):
            start = frame_idx * 4
            chunk = raw_lines[start:start + 4]
            frame_id, channels = self._parse_frame_chunk(meta.path, chunk, frame_idx, start + 1)
            if channels is None:
                if self.strict:
                    latest = self.issues[-1].message if self.issues else 'Unknown frame parsing error'
                    raise DataFormatError(
                        f'{meta.path}: frame_index={frame_idx}, line={start + 1}, error={latest}'
                    )
                continue

            depth = channels[0].shape[0]
            for ch_data in channels[1:]:
                if ch_data.shape[0] != depth:
                    self._record_issue(
                        meta.path,
                        frame_idx,
                        start + 1,
                        f'Inconsistent depth in frame {frame_id}: {ch_data.shape[0]} vs {depth}'
                    )
                    if self.strict:
                        return None
                    depth = -1
                    break
            if depth < 0:
                continue

            if self.global_depth is None:
                self.global_depth = depth
            elif depth != self.global_depth:
                self._record_issue(
                    meta.path,
                    frame_idx,
                    start + 1,
                    f'Global depth mismatch: got {depth}, expected {self.global_depth}'
                )
                if self.strict:
                    return None
                continue

            frame_array = np.stack(channels, axis=0)  # (4, depth)
            frame_list.append(frame_array)

        if not frame_list:
            return self._fail_or_skip(meta.path, -1, -1, 'No valid frames after parsing')

        return ParsedRecord(meta=meta, frames=np.stack(frame_list, axis=0))

    def _parse_frame_chunk(
        self,
        path: str,
        chunk: List[str],
        frame_idx: int,
        line_start: int,
    ) -> Tuple[Optional[str], Optional[List[np.ndarray]]]:
        parsed_rows: List[Tuple[str, str, np.ndarray, int]] = []

        for offset, line in enumerate(chunk):
            line_no = line_start + offset
            parts = line.split()
            if len(parts) < 3:
                self._record_issue(path, frame_idx, line_no, f'Bad line format: {line}')
                if self.strict:
                    return None, None
                return None, None

            frame_id = parts[0]
            channel = parts[1]
            if not frame_id.startswith('F'):
                self._record_issue(path, frame_idx, line_no, f'Invalid frame id: {frame_id}')
                if self.strict:
                    return None, None
                return None, None

            channel_match = _CHANNEL_RE.match(channel)
            if channel_match is None:
                self._record_issue(path, frame_idx, line_no, f'Invalid channel id: {channel}')
                if self.strict:
                    return None, None
                return None, None

            try:
                samples = np.asarray([float(v) for v in parts[2:]], dtype=np.float32)
            except ValueError:
                self._record_issue(path, frame_idx, line_no, 'Non-numeric sample value found')
                if self.strict:
                    return None, None
                return None, None

            parsed_rows.append((frame_id, channel.upper(), samples, int(channel_match.group('index'))))

        frame_ids = {row[0] for row in parsed_rows}
        if len(frame_ids) != 1:
            self._record_issue(path, frame_idx, line_start, f'Frame id mismatch in one frame chunk: {frame_ids}')
            if self.strict:
                return None, None
            return None, None

        channel_ids = [row[1] for row in parsed_rows]
        if len(set(channel_ids)) != 4:
            self._record_issue(path, frame_idx, line_start, f'Channel count/duplication error: {channel_ids}')
            if self.strict:
                return None, None
            return None, None

        parsed_rows.sort(key=lambda item: item[3])
        channels = [row[2] for row in parsed_rows]
        return parsed_rows[0][0], channels

    def _record_issue(self, path: str, frame_idx: int, line_no: int, message: str) -> None:
        issue = ParseIssue(path=path, frame_index=frame_idx, line_number=line_no, message=message)
        self.issues.append(issue)

    def _fail_or_skip(
        self,
        path: str,
        frame_idx: int,
        line_no: int,
        message: str,
    ) -> Optional[ParsedRecord]:
        self._record_issue(path, frame_idx, line_no, message)
        if self.strict:
            raise DataFormatError(f'{path}: {message}')
        return None


class FlatUltrasoundDataset(TorchDataset):  # type: ignore[misc]
    """Dataset for flat ultrasound files in 20260416-like format."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        split_config: Optional[SplitConfig] = None,
        strict: bool = True,
        preprocess_mode: str = 'standardize_only',
        flatten_for_lstm: bool = True,
        known_gesture_order: Optional[Sequence[str]] = None,
        label_dtype: Any = None,
    ):
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        if not _TORCH_AVAILABLE:
            raise ImportError('torch is required to build FlatUltrasoundDataset')

        self.root_dir = root_dir
        self.split = split
        self.strict = strict
        self.preprocess_mode = preprocess_mode
        self.flatten_for_lstm = flatten_for_lstm
        self.label_dtype = label_dtype if label_dtype is not None else getattr(torch, 'float32')
        self.split_config = split_config or SplitConfig()

        parser = FlatUltrasoundParser(strict=strict)
        metas = parser.discover(root_dir)
        records = []
        for meta in metas:
            parsed = parser.parse_file(meta)
            if parsed is not None:
                records.append(parsed)

        if not records:
            raise DataFormatError('No valid files were parsed from flat dataset directory')

        allowed_order = list(known_gesture_order) if known_gesture_order is not None else list(DEFAULT_GESTURE_ORDER)
        appeared = {record.meta.gesture for record in records}
        self.classes = [gesture for gesture in allowed_order if gesture in appeared]

        if not self.classes:
            raise DataFormatError('No recognized gestures found after filtering')

        unknown_gestures = sorted(appeared - set(self.classes))
        if unknown_gestures:
            msg = f'Unknown gestures found and ignored: {unknown_gestures}'
            if strict:
                raise DataFormatError(msg)
            parser._record_issue(path=root_dir, frame_idx=-1, line_no=-1, message=msg)
            records = [r for r in records if r.meta.gesture in set(self.classes)]

        self.class_to_idx = {gesture: idx for idx, gesture in enumerate(self.classes)}

        strategy = self.split_config.strategy.lower()
        if strategy == 'ratio':
            # ratio 模式改为“文件内按帧比例切分”，因此每个 split 都遍历全部文件。
            selected_records = records
        else:
            split_map = split_file_records(records, self.split_config)
            selected_records = split_map[split]
            if not selected_records:
                raise DataFormatError(f'Split {split} has no data')

        image_list: List[np.ndarray] = []
        label_list: List[int] = []
        selected_file_count = 0

        for record in selected_records:
            data = record.frames.transpose(1, 2, 0)  # (4, depth, frames)
            data = Preprocessing(data, mode=self.preprocess_mode)
            data = data.transpose(2, 0, 1)  # (frames, 4, depth)

            if strategy == 'ratio':
                frame_indices = split_frame_indices_by_ratio(
                    num_frames=int(data.shape[0]),
                    split=split,
                    config=self.split_config,
                    file_key=record.meta.path,
                )
                if not frame_indices:
                    continue
                frames_iter = (data[idx] for idx in frame_indices)
            else:
                frames_iter = (frame for frame in data)

            selected_file_count += 1

            label = self.class_to_idx[record.meta.gesture]
            for frame in frames_iter:
                if self.flatten_for_lstm:
                    image_list.append(frame.reshape(1, -1).astype(np.float32))
                else:
                    image_list.append(frame[np.newaxis, :, :].astype(np.float32))
                label_list.append(label)

        if not image_list:
            raise DataFormatError(f'Split {split} has no data')

        self.images = getattr(torch, 'from_numpy')(np.stack(image_list, axis=0))
        self.labels = getattr(torch, 'tensor')(label_list, dtype=label_dtype)

        self.report = {
            'valid_files': len(records),
            'selected_files': selected_file_count,
            'ignored_files': len(parser.ignored_files),
            'issues': len(parser.issues),
            'sampling_depth': parser.global_depth,
            'classes': self.classes,
            'class_count': len(self.classes),
            'split': split,
            'samples': int(self.images.shape[0]),
        }
        self.issues = parser.issues

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def split_file_records(records: Sequence[ParsedRecord], config: SplitConfig) -> Dict[str, List[ParsedRecord]]:
    strategy = config.strategy.lower()

    if strategy == 'trial':
        train_trials = set(config.train_trials)
        val_trials = set(config.val_trials)
        test_trials = set(config.test_trials)
        split_map = {'train': [], 'val': [], 'test': []}
        for record in records:
            trial = record.meta.trial
            if trial in train_trials:
                split_map['train'].append(record)
            elif trial in val_trials:
                split_map['val'].append(record)
            elif trial in test_trials:
                split_map['test'].append(record)
        return split_map

    if strategy == 'explicit':
        if not config.explicit_map:
            raise ValueError('explicit_map is required when strategy is explicit')
        split_map = {'train': [], 'val': [], 'test': []}
        for record in records:
            key = f'{record.meta.subject}-{record.meta.gesture}-{record.meta.trial}'
            target = config.explicit_map.get(key)
            if target not in {'train', 'val', 'test'}:
                continue
            split_map[target].append(record)
        return split_map

    if strategy == 'ratio':
        total = len(records)
        if total == 0:
            return {'train': [], 'val': [], 'test': []}

        if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-6:
            raise ValueError('train_ratio + val_ratio + test_ratio must be 1.0')

        shuffled = list(records)
        rng = random.Random(config.seed)
        rng.shuffle(shuffled)

        train_end = int(total * config.train_ratio)
        val_end = train_end + int(total * config.val_ratio)

        return {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:],
        }

    raise ValueError("Unsupported split strategy. Use 'ratio', 'trial', or 'explicit'.")


def split_frame_indices_by_ratio(
    num_frames: int,
    split: str,
    config: SplitConfig,
    file_key: str,
) -> List[int]:
    if num_frames <= 0:
        return []

    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio must be 1.0')

    indices = list(range(num_frames))
    # 使用“全局 seed + 文件路径”保证三个 split 构造时得到同一帧划分边界。
    rng = random.Random(f'{config.seed}:{file_key}')
    rng.shuffle(indices)

    train_end = int(num_frames * config.train_ratio)
    val_end = train_end + int(num_frames * config.val_ratio)

    if split == 'train':
        return indices[:train_end]
    if split == 'val':
        return indices[train_end:val_end]
    if split == 'test':
        return indices[val_end:]

    raise ValueError("split must be one of: 'train', 'val', 'test'")


def build_flat_datasets(
    root_dir: str,
    split_config: Optional[SplitConfig] = None,
    strict: bool = True,
    preprocess_mode: str = 'standardize_only',
    flatten_for_lstm: bool = True,
    known_gesture_order: Optional[Sequence[str]] = None,
    label_dtype: Any = None,
) -> Dict[str, FlatUltrasoundDataset]:
    cfg = split_config or SplitConfig()
    return {
        'train': FlatUltrasoundDataset(
            root_dir=root_dir,
            split='train',
            split_config=cfg,
            strict=strict,
            preprocess_mode=preprocess_mode,
            flatten_for_lstm=flatten_for_lstm,
            known_gesture_order=known_gesture_order,
            label_dtype=label_dtype,
        ),
        'val': FlatUltrasoundDataset(
            root_dir=root_dir,
            split='val',
            split_config=cfg,
            strict=strict,
            preprocess_mode=preprocess_mode,
            flatten_for_lstm=flatten_for_lstm,
            known_gesture_order=known_gesture_order,
            label_dtype=label_dtype,
        ),
        'test': FlatUltrasoundDataset(
            root_dir=root_dir,
            split='test',
            split_config=cfg,
            strict=strict,
            preprocess_mode=preprocess_mode,
            flatten_for_lstm=flatten_for_lstm,
            known_gesture_order=known_gesture_order,
            label_dtype=label_dtype,
        ),
    }
