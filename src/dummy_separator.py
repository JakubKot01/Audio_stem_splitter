import numpy as np

from src.separator_base import BaseSeparator


class DummySeparator(BaseSeparator):
    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        zeros = np.zeros_like(audio)

        return {
            "vocals": audio.copy(),
            "drums": zeros.copy(),
            "bass": zeros.copy(),
            "other": zeros.copy(),
        }