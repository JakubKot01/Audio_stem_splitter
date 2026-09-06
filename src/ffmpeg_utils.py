from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_ffmpeg_available(model_file_dir: Path) -> Path:
    """Make either system FFmpeg or imageio's bundled binary available by name."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return Path(system_ffmpeg)

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "FFmpeg is required by audio-separator. Install it in .venv-mdx "
            "with: python -m pip install imageio-ffmpeg"
        ) from error

    bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not bundled_ffmpeg.is_file():
        raise RuntimeError(
            f"imageio-ffmpeg did not provide a usable binary: {bundled_ffmpeg}"
        )

    ffmpeg_dir = model_file_dir / "ffmpeg"
    local_ffmpeg = ffmpeg_dir / "ffmpeg.exe"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    if not local_ffmpeg.is_file() or local_ffmpeg.stat().st_size != bundled_ffmpeg.stat().st_size:
        shutil.copy2(bundled_ffmpeg, local_ffmpeg)

    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    return local_ffmpeg
