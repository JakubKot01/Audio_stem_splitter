from abc import ABC, abstractmethod
import numpy as np


class BaseSeparator(ABC):
    @abstractmethod
    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Returns:
            dict mapping stem name -> separated audio array with shape (channels, samples)
        """
        raise NotImplementedError