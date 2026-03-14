import numpy as np
import librosa


def load_audio(file_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file_path, sr=target_sr, mono=False)

    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)

    return audio.astype(np.float32), sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak