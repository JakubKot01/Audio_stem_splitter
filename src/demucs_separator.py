from pathlib import Path
from time import perf_counter

import demucs.separate


class DemucsSeparator:
    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda",
        segment: int | None = None,
        shifts: int = 1,
        mp3: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.segment = segment
        self.shifts = shifts
        self.mp3 = mp3

    def separate_file(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "-n", self.model_name,
            "-d", self.device,
            "--shifts", str(self.shifts),
            "-o", str(output_dir),
            str(input_path),
        ]

        if self.segment is not None:
            args.extend(["--segment", str(self.segment)])

        if self.mp3:
            args.append("--mp3")

        start_time = perf_counter()
        demucs.separate.main(args)
        elapsed = perf_counter() - start_time

        stems_dir = output_dir / self.model_name / input_path.stem

        return {
            "input_file": str(input_path),
            "stems_dir": str(stems_dir),
            "model_name": self.model_name,
            "device": self.device,
            "elapsed_seconds": elapsed,
        }