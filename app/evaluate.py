import json
from pathlib import Path

import typer

from src.evaluation import (
    build_evaluation_history_row,
    evaluate_stem_directories,
    load_stem_mapping,
)
from src.experiment_utils import append_experiment_to_csv, save_json


app = typer.Typer(help="Evaluate estimated stems against known reference stems")


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} dB"


@app.command()
def evaluate(
    estimates_dir: str = typer.Argument(..., help="Directory containing estimated stems"),
    references_dir: str = typer.Argument(..., help="Directory containing reference stems"),
    mixture_file: str = typer.Option(
        "",
        help="Original mixture; enables SI-SDR improvement (SI-SDRi)",
    ),
    mapping_file: str = typer.Option(
        "",
        help="JSON mapping from reference stems to estimated stems",
    ),
    output_file: str = typer.Option(
        "evaluation.json",
        help="Destination JSON file",
    ),
    history_file: str = typer.Option(
        "experiments/evaluation_history.csv",
        help="CSV collecting results across evaluated runs; empty disables it",
    ),
    max_length_mismatch: float = typer.Option(
        1.0,
        min=0.0,
        help="Maximum permitted duration mismatch in seconds",
    ),
):
    estimates_path = Path(estimates_dir)
    mapping = load_stem_mapping(Path(mapping_file)) if mapping_file else None
    result = evaluate_stem_directories(
        estimates_dir=estimates_path,
        references_dir=Path(references_dir),
        mixture_file=Path(mixture_file) if mixture_file else None,
        stem_mapping=mapping,
        max_length_mismatch_seconds=max_length_mismatch,
    )

    metadata_path = estimates_path.parent / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        result["run"] = {
            key: metadata.get(key)
            for key in (
                "run_name",
                "model_name",
                "device",
                "segment",
                "shifts",
                "mp3",
                "input_file",
            )
        }
    else:
        result["run"] = {
            "run_name": estimates_path.parent.name,
            "input_file": mixture_file or None,
        }

    output_path = Path(output_file)
    save_json(result, output_path)
    if history_file:
        append_experiment_to_csv(
            build_evaluation_history_row(result, output_path),
            Path(history_file),
        )

    summary = result["summary"]
    typer.echo(f"Evaluated stems: {', '.join(summary['evaluated_stems'])}")
    typer.echo(f"Mean SI-SDR: {_format_metric(summary['mean_si_sdr_db'])}")
    typer.echo(f"Mean SDR: {_format_metric(summary['mean_sdr_db'])}")
    typer.echo(f"Mean SI-SDRi: {_format_metric(summary['mean_si_sdri_db'])}")
    typer.echo(f"Detailed results: {output_file}")
    if history_file:
        typer.echo(f"Comparison history: {history_file}")

    if summary["missing_estimates"]:
        typer.echo(f"Missing estimates: {summary['missing_estimates']}", err=True)
    if summary["unused_estimates"]:
        typer.echo(f"Unused estimates: {summary['unused_estimates']}")


if __name__ == "__main__":
    app()
