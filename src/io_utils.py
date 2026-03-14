from pathlib import Path

import soundfile as sf


def save_audio(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate)


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")


def ensure_supported_extension(path: Path, supported_extensions: set[str]) -> None:
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file extension: {path.suffix}. "
            f"Supported: {sorted(supported_extensions)}"
        )