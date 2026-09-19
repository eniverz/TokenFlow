"""A weighted mixture of PyTorch datasets.

Each entry in ``datasets`` is a sampling group.  The group weight controls its
logical length when weighted sampling is enabled, matching GalaxeaVLA's
``MixtureLerobotDataset`` semantics.
"""

import math
from typing import List, Optional, Sequence, Tuple

from torch.utils.data.dataset import Dataset

from resVLA.tools.overwatch import initialize_overwatch
from .transform_dataset import TransformDataset

logger = initialize_overwatch(__name__)


def normalize_dataset_weights(lengths: Sequence[int], weights: Sequence[float]) -> List[float]:
    """Normalize weights while preserving the total logical length."""
    if len(lengths) != len(weights):
        raise ValueError("lengths and weights must have the same length")
    total_length = sum(int(length) for length in lengths)
    weighted_length = sum(
        int(length) * float(weight) for length, weight in zip(lengths, weights)
    )
    if weighted_length <= 0:
        raise ValueError("Invalid weight normalization denominator (<= 0)")
    scale = total_length / weighted_length
    return [scale * float(weight) for weight in weights]


class MixtureDataset(Dataset):
    def __init__(
        self,
        datasets: List[TransformDataset],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if not datasets:
            raise ValueError("MixtureDataset requires at least one dataset")
        self.datasets = list(datasets)
        self.actual_lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in self.actual_lengths):
            raise ValueError(f"All datasets must be non-empty, got lengths={self.actual_lengths}")

        if weights is None:
            weights = [1.0] * len(self.datasets)
        if len(weights) != len(self.datasets):
            raise ValueError(
                f"weights and datasets must have the same length, got "
                f"{len(weights)} and {len(self.datasets)}"
            )
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
            raise ValueError(f"weights must be finite and non-negative, got {weights}")

        self.weights = normalize_dataset_weights(self.actual_lengths, weights)
        self.sampling_epoch = 0

        self.effective_lengths = self._compute_effective_lengths()
        self.effective_starts, self.effective_ends = self._set_effective_offsets(self.effective_lengths)

        # Backwards-compatible public fields.
        self.lengths = list(self.actual_lengths)
        self.actual_total_length = sum(self.actual_lengths)
        self.total_length = int(self.effective_ends[-1])

    def _compute_effective_lengths(self) -> List[int]:
        effective_lengths = []
        for dataset_idx, (actual_length, weight) in enumerate(
            zip(self.actual_lengths, self.weights)
        ):
            if weight <= 0:
                raise ValueError(
                    "use_weight_for_sampling=True requires positive weights; "
                    f"dataset {dataset_idx} has weight={weight}"
                )
            effective_lengths.append(max(1, int(actual_length * weight)))
        return effective_lengths

    def _set_effective_offsets(self, lengths: Sequence[int]) -> Tuple[List[int], List[int]]:
        # Lists keep the dataset easy to pickle in DataLoader workers.
        effective_starts = []
        effective_ends = []
        offset = 0
        for length in lengths:
            effective_starts.append(offset)
            offset += int(length)
            effective_ends.append(offset)

        return effective_starts, effective_ends

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic weighted undersampling."""
        self.sampling_epoch = int(epoch)

    def _map_weighted_local_index(self, dataset_idx: int, local_idx: int) -> int:
        actual_length = self.actual_lengths[dataset_idx]
        effective_length = self.effective_lengths[dataset_idx]

        if effective_length >= actual_length:
            return local_idx % actual_length

        # Spread undersampled indices across the source; change the subset per
        # epoch without introducing nondeterminism in workers.
        stride = max(1, int(actual_length * 0.61803398875))
        while math.gcd(stride, actual_length) != 1:
            stride += 1
            if stride >= actual_length:
                stride = 1
                break
        offset = (self.sampling_epoch * 1_000_003 + dataset_idx) % actual_length
        return int((local_idx * stride + offset) % actual_length)

    def _resolve_index(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for mixture of length {len(self)}")
        for dataset_idx, end in enumerate(self.effective_ends):
            if index < end:
                local_idx = index - self.effective_starts[dataset_idx]
                return dataset_idx, self._map_weighted_local_index(dataset_idx, local_idx)
        raise IndexError(f"Index {index} out of bounds for mixture of length {len(self)}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        dataset_idx, local_idx = self._resolve_index(int(index))
        return self.datasets[dataset_idx][local_idx]
