from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class DataSample:
    task_id: str
    task: str
    ground_truth: str
    metadata: dict = field(default_factory=dict)
    domain: str = ""


class BaseDataset(ABC):
    """Base class for all benchmark datasets."""

    def __init__(self, split: str = "test", limit: Optional[int] = None, data_path: Optional[str] = None):
        self.split = split
        self.limit = limit
        self.data_path = data_path
        self._samples: Optional[List[DataSample]] = None

    @property
    def samples(self) -> List[DataSample]:
        if self._samples is None:
            self._samples = self._load()
            if self.limit:
                self._samples = self._samples[:self.limit]
        return self._samples

    @abstractmethod
    def _load(self) -> List[DataSample]:
        pass

    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str) -> float:
        """Return accuracy score in [0, 1]."""
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)
