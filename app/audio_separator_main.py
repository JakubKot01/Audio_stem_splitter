from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from src.config import DEFAULT_EXPERIMENTS_DIR, DEFAULT_OUTPUT_DIR, SUPPORTED_EXTENSIONS
from src.evaluation import (
    build_evaluation_history_row,
    evaluate_stem_directories,
    load_stem_mapping,
)
from src.experiment_utils import append_experiment_to_csv, generate_run_name, save_json
from src.ffmpeg_utils import ensure_ffmpeg_available
from src.io_utils import ensure_file_exists, ensure_supported_extension
from src.logger_utils import setup_logger
from src.model_catalog import get_model_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an audio-separator MDX/MDXC two-stem model."
    )
    parser.add_argument("input_file")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-file-dir", default="models/audio-separator")
    parser.add_argument("--mp3", action="store_true")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--evaluation-mapping", default="")
    parser.add_argument("--max-length-mismatch", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = Path(args.input_file)
    output_root = Path(args.output_dir)
    experiments_root = Path(args.experiments_dir)
    model_spec = get_model_spec(args.model_name)
    if model_spec.backend != "audio-separator":
        raise ValueError(f"{args.model_name} is not an audio-separator model")

    ensure_file_exists(input_path)
    ensure_supported_extension(input_path, SUPPORTED_EXTENSIONS)
    if args.evaluation_mapping and not args.reference_dir:
        raise ValueError("--evaluation-mapping requires --reference-dir")

    references_path = Path(args.reference_dir) if args.reference_dir else None
    if references_path is not None and not references_path.is_dir():
        raise NotADirectoryError(
            f"Reference directory does not exist: {references_path}"
        )
    mapping_path = Path(args.evaluation_mapping) if args.evaluation_mapping else None
    mapping = load_stem_mapping(mapping_path) if mapping_path else None

    run_name = generate_run_name(input_path, args.run_name or None)
    run_output_dir = output_root / args.model_name / run_name
    stems_dir = run_output_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_output_dir / "logs.txt")

    logger.info("Starting audio-separator separation")
    logger.info("Input file: %s", input_path)
    logger.info("Run name: %s", run_name)
    logger.info("Model: %s", args.model_name)
    logger.info("Purpose: %s", model_spec.purpose)
    logger.info("The model is downloaded automatically on first use.")

    ffmpeg_path = ensure_ffmpeg_available(Path(args.model_file_dir))
    logger.info("FFmpeg: %s", ffmpeg_path)

    try:
        from audio_separator.separator import Separator
    except ImportError as error:
        raise RuntimeError(
            "audio-separator is not installed in the MDX environment. Install "
            "audio-separator[cpu] or audio-separator[gpu] in .venv-mdx."
        ) from error

    separator = Separator(
        log_level=logging.INFO,
        model_file_dir=str(Path(args.model_file_dir)),
        output_dir=str(stems_dir),
        output_format="MP3" if args.mp3 else "WAV",
        normalization_threshold=1.0,
        amplification_threshold=0.0,
    )
    started = perf_counter()
    separator.load_model(model_filename=args.model_name)
    output_files = separator.separate(
        str(input_path),
        {
            "Vocals": "vocals",
            "Instrumental": "instrumental",
            # A few vocal checkpoints call their complementary output Other.
            "Other": "instrumental",
        },
    )
    elapsed = perf_counter() - started

    metadata = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "original_input_name": input_path.name,
        "model_name": args.model_name,
        "backend": "audio-separator",
        "architecture": model_spec.architecture,
        "purpose": model_spec.purpose,
        "device": "automatic",
        "segment": None,
        "shifts": None,
        "mp3": args.mp3,
        "output_dir": str(run_output_dir),
        "stems_dir": str(stems_dir),
        "output_files": [str(path) for path in output_files],
        "elapsed_seconds": round(elapsed, 4),
        "evaluation_status": "not_requested",
    }

    evaluation_error: Exception | None = None
    if references_path is not None:
        logger.info("Evaluating matching stems against: %s", references_path)
        logger.info(
            "With standard MUSDB four-stem references, this two-stem run is "
            "evaluated on vocals; instrumental needs an aggregated reference."
        )
        try:
            evaluation = evaluate_stem_directories(
                estimates_dir=stems_dir,
                references_dir=references_path,
                mixture_file=input_path,
                stem_mapping=mapping,
                max_length_mismatch_seconds=args.max_length_mismatch,
            )
            evaluation["run"] = {
                "run_name": run_name,
                "model_name": args.model_name,
                "backend": "audio-separator",
                "device": "automatic",
                "segment": None,
                "shifts": None,
                "mp3": args.mp3,
                "input_file": str(input_path),
            }
            run_evaluation_path = run_output_dir / "evaluation.json"
            archived_evaluation_path = (
                experiments_root / "evaluations" / f"{run_name}.json"
            )
            save_json(evaluation, run_evaluation_path)
            save_json(evaluation, archived_evaluation_path)
            append_experiment_to_csv(
                build_evaluation_history_row(evaluation, archived_evaluation_path),
                experiments_root / "evaluation_history.csv",
            )
            summary = evaluation["summary"]
            metadata.update(
                {
                    "evaluation_status": "completed",
                    "reference_dir": str(references_path),
                    "evaluation_mapping": str(mapping_path) if mapping_path else "identity",
                    "evaluation_json": str(archived_evaluation_path),
                    "evaluated_stems": summary["evaluated_stems"],
                    "mean_si_sdr_db": summary["mean_si_sdr_db"],
                    "median_si_sdr_db": summary["median_si_sdr_db"],
                    "mean_sdr_db": summary["mean_sdr_db"],
                    "median_sdr_db": summary["median_sdr_db"],
                    "mean_si_sdri_db": summary["mean_si_sdri_db"],
                    "median_si_sdri_db": summary["median_si_sdri_db"],
                }
            )
        except Exception as error:
            evaluation_error = error
            metadata.update(
                {
                    "evaluation_status": "failed",
                    "reference_dir": str(references_path),
                    "evaluation_error": str(error),
                }
            )
            logger.exception("Evaluation failed after separation completed")

    save_json(metadata, run_output_dir / "metadata.json")
    save_json(metadata, experiments_root / f"{run_name}.json")
    append_experiment_to_csv(metadata, experiments_root / "experiment_history.csv")

    logger.info("Separation finished in %.4f seconds", elapsed)
    logger.info("Stems saved to: %s", stems_dir)
    print("Separation finished.")
    print(f"Run name: {run_name}")
    print(f"Stems dir: {stems_dir}")

    if evaluation_error is not None:
        print(f"Separation succeeded, but evaluation failed: {evaluation_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
