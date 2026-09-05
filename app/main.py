from datetime import datetime
from pathlib import Path
import shutil

import typer

from src.config import (
    DEFAULT_EXPERIMENTS_DIR,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.demucs_separator import DemucsSeparator
from src.device import get_default_device
from src.evaluation import (
    build_evaluation_history_row,
    evaluate_stem_directories,
    load_stem_mapping,
)
from src.experiment_utils import (
    append_experiment_to_csv,
    generate_run_name,
    save_json,
)
from src.io_utils import ensure_file_exists, ensure_supported_extension
from src.logger_utils import setup_logger

app = typer.Typer(help="Audio source separation CLI")


def _format_db(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} dB"


@app.command()
def separate(
    input_file: str = typer.Argument(..., help="Path to audio file"),
    output_dir: str = typer.Option(str(DEFAULT_OUTPUT_DIR), help="Output directory"),
    experiments_dir: str = typer.Option(str(DEFAULT_EXPERIMENTS_DIR), help="Experiments directory"),
    model_name: str = typer.Option("htdemucs", help="Demucs model name"),
    device: str = typer.Option("", help="Device: cuda or cpu"),
    segment: int = typer.Option(0, help="Segment length in seconds; 0 = default"),
    shifts: int = typer.Option(1, help="Number of shifts"),
    mp3: bool = typer.Option(False, help="Save output as mp3 instead of wav"),
    run_name: str = typer.Option("", help="Custom experiment/run name"),
    reference_dir: str = typer.Option(
        "",
        help="Known reference stems; enables objective evaluation",
    ),
    evaluation_mapping: str = typer.Option(
        "",
        help="JSON mapping from reference stems to estimated stems",
    ),
    max_length_mismatch: float = typer.Option(
        1.0,
        min=0.0,
        help="Maximum permitted evaluation duration mismatch in seconds",
    ),
):
    input_path = Path(input_file)
    output_root = Path(output_dir)
    experiments_root = Path(experiments_dir)

    ensure_file_exists(input_path)
    ensure_supported_extension(input_path, SUPPORTED_EXTENSIONS)

    if evaluation_mapping and not reference_dir:
        raise typer.BadParameter(
            "--evaluation-mapping requires --reference-dir"
        )

    references_path = Path(reference_dir) if reference_dir else None
    mapping_path = Path(evaluation_mapping) if evaluation_mapping else None
    resolved_evaluation_mapping = None
    if references_path is not None:
        if not references_path.is_dir():
            raise typer.BadParameter(
                f"Reference directory does not exist: {references_path}"
            )
        resolved_evaluation_mapping = (
            load_stem_mapping(mapping_path) if mapping_path else None
        )

    resolved_device = device if device else get_default_device()
    resolved_segment = None if segment == 0 else segment
    resolved_run_name = generate_run_name(input_path, run_name if run_name else None)

    run_output_dir = output_root / model_name / resolved_run_name
    log_file = run_output_dir / "logs.txt"
    logger = setup_logger(log_file)

    logger.info("Starting separation")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Run name: {resolved_run_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {resolved_device}")

    separator = DemucsSeparator(
        model_name=model_name,
        device=resolved_device,
        segment=resolved_segment,
        shifts=shifts,
        mp3=mp3,
    )

    result = separator.separate_file(
        input_path=input_path,
        output_dir=run_output_dir,
    )

    demucs_stems_dir = Path(result["stems_dir"])
    final_stems_dir = run_output_dir / "stems"
    final_stems_dir.mkdir(parents=True, exist_ok=True)

    for stem_file in demucs_stems_dir.glob("*"):
        shutil.move(str(stem_file), final_stems_dir / stem_file.name)

    metadata = {
        "run_name": resolved_run_name,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "original_input_name": input_path.name,
        "model_name": model_name,
        "device": resolved_device,
        "segment": resolved_segment,
        "shifts": shifts,
        "mp3": mp3,
        "output_dir": str(run_output_dir),
        "stems_dir": str(final_stems_dir),
        "elapsed_seconds": round(result["elapsed_seconds"], 4),
        "evaluation_status": "not_requested",
    }

    evaluation_error: Exception | None = None
    if references_path is not None:
        logger.info(f"Evaluating stems against references: {references_path}")
        try:
            evaluation = evaluate_stem_directories(
                estimates_dir=final_stems_dir,
                references_dir=references_path,
                mixture_file=input_path,
                stem_mapping=resolved_evaluation_mapping,
                max_length_mismatch_seconds=max_length_mismatch,
            )
            evaluation["run"] = {
                "run_name": resolved_run_name,
                "model_name": model_name,
                "device": resolved_device,
                "segment": resolved_segment,
                "shifts": shifts,
                "mp3": mp3,
                "input_file": str(input_path),
            }

            run_evaluation_path = run_output_dir / "evaluation.json"
            archived_evaluation_path = (
                experiments_root / "evaluations" / f"{resolved_run_name}.json"
            )
            save_json(evaluation, run_evaluation_path)
            save_json(evaluation, archived_evaluation_path)
            append_experiment_to_csv(
                build_evaluation_history_row(
                    evaluation,
                    archived_evaluation_path,
                ),
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
            logger.info(
                "Evaluation finished: mean SI-SDR=%s dB, mean SDR=%s dB, "
                "mean SI-SDRi=%s dB",
                summary["mean_si_sdr_db"],
                summary["mean_sdr_db"],
                summary["mean_si_sdri_db"],
            )
        except Exception as error:
            evaluation_error = error
            metadata.update(
                {
                    "evaluation_status": "failed",
                    "reference_dir": str(references_path),
                    "evaluation_mapping": str(mapping_path) if mapping_path else "identity",
                    "evaluation_error": str(error),
                }
            )
            logger.exception("Evaluation failed after separation completed")

    save_json(metadata, run_output_dir / "metadata.json")
    save_json(metadata, experiments_root / f"{resolved_run_name}.json")
    append_experiment_to_csv(metadata, experiments_root / "experiment_history.csv")

    logger.info("Separation finished successfully")
    logger.info(f"Elapsed time: {metadata['elapsed_seconds']} s")
    logger.info(f"Results saved to: {run_output_dir}")

    typer.echo("Separation finished.")
    typer.echo(f"Run name: {resolved_run_name}")
    typer.echo(f"Input: {metadata['input_file']}")
    typer.echo(f"Output dir: {metadata['output_dir']}")
    typer.echo(f"Stems dir: {metadata['stems_dir']}")
    typer.echo(f"Elapsed time: {metadata['elapsed_seconds']} s")

    if metadata["evaluation_status"] == "completed":
        typer.echo(f"Mean SI-SDR: {_format_db(metadata['mean_si_sdr_db'])}")
        typer.echo(f"Mean SDR: {_format_db(metadata['mean_sdr_db'])}")
        typer.echo(f"Mean SI-SDRi: {_format_db(metadata['mean_si_sdri_db'])}")
        typer.echo(f"Evaluation: {metadata['evaluation_json']}")

    if evaluation_error is not None:
        typer.echo(
            "Separation succeeded, but evaluation failed: "
            f"{evaluation_error}",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
