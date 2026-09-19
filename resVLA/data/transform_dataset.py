from typing import List, Literal

from torch.utils.data.dataset import Dataset

from resVLA.tools.overwatch import initialize_overwatch
from .transform import BaseTransform

logger = initialize_overwatch(__name__)


class TransformDataset(Dataset):
    def __init__(
        self,
        repo_ids: List[str],
        *,
        lerobot_version: Literal["v2", "v3"],
        action_horizon: float,
        action_only: bool,
        video_backend: Literal["pyav", "torchcodec"],
        transform: List[BaseTransform],
    ):
        super().__init__()

        self.datasets = [self._create_lerobot_dataset(
            repo_id=repo_id,
            lerobot_version=lerobot_version,
            action_horizon=action_horizon,
            action_only=action_only,
            video_backend=video_backend
        ) for repo_id in repo_ids]
        
        self.transform = transform
        self.all_lengths = [len(ds) for ds in self.datasets]
        self.length = sum(self.all_lengths)

    def _create_lerobot_dataset(
        self,
        repo_id: str,
        lerobot_version: Literal["v2.1", "v3.0"],
        action_horizon: float,
        action_only: bool,
        video_backend: Literal["pyav", "torchcodec"]
    ):
        if lerobot_version == "v2.1":
            from resVLA.data.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

            DatasetClass = LeRobotDataset
            MetaClass = LeRobotDatasetMetadata
        elif lerobot_version == "v3.0":
            from resVLA.data.lerobot.lerobot_dataset_v3 import LeRobotDataset, LeRobotDatasetMetadata

            DatasetClass = LeRobotDataset
            MetaClass = LeRobotDatasetMetadata
        else:
            raise ValueError(f"Unsupported lerobot_version: {lerobot_version}")

        meta = MetaClass(repo_id)
        fps = meta.fps
        return DatasetClass(
            repo_id=repo_id,
            video_backend=video_backend,
            load_images=not action_only,
            delta_timestamps={
                "action": [t / fps for t in range(action_horizon)]
            }
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ds_idx = 0
        while index >= self.all_lengths[ds_idx]:
            index -= self.all_lengths[ds_idx]
            ds_idx += 1
        ds_len = self.all_lengths[ds_idx]
        last_error = None

        for offset in range(ds_len):
            idx = (index + offset) % ds_len
            try:
                data = self.datasets[ds_idx][idx]
                for t in self.transform:
                    data = t(data)
                return data
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to load sample at index {idx} (dataset: {ds_idx}, sample: {index});"
                    f"trying next sample.\nError: {type(e).__name__}: {e}."
                )

        raise RuntimeError(
            f"Failed to load any sample after {self.length} attempts" 
            f"starting from index {index}"
        ) from last_error
