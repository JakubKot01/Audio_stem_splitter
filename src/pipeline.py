from pathlib import Path
from time import perf_counter

from src.config import DEFAULT_OUTPUT_DIR
from src.io_utils import save_audio
from src.preprocess import load_audio, normalize_audio


def run_separation_pipeline(
    input_path: Path,
    separator,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_rate: int = 44100,
) -> dict:
    start_time = perf_counter()

    audio, sr = load_audio(str(input_path), target_sr=sample_rate)
    audio = normalize_audio(audio)

    stems = separator.separate(audio, sr)

    song_output_dir = output_dir / input_path.stem
    song_output_dir.mkdir(parents=True, exist_ok=True)

    for stem_name, stem_audio in stems.items():
        output_path = song_output_dir / f"{stem_name}.wav"
        save_audio(output_path, stem_audio, sr)

    elapsed = perf_counter() - start_time

    return {
        "input_file": str(input_path),
        "output_dir": str(song_output_dir),
        "sample_rate": sr,
        "stems": list(stems.keys()),
        "elapsed_seconds": elapsed,
    }