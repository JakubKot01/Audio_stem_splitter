from pathlib import Path

import typer

from src.config import DEFAULT_OUTPUT_DIR, DEFAULT_SAMPLE_RATE, SUPPORTED_EXTENSIONS
from src.dummy_separator import DummySeparator
from src.io_utils import ensure_file_exists, ensure_supported_extension
from src.pipeline import run_separation_pipeline

app = typer.Typer(help="Audio source separation CLI")


@app.command()
def separate(
    input_file: str = typer.Argument(..., help="Path to audio file"),
    output_dir: str = typer.Option(str(DEFAULT_OUTPUT_DIR), help="Output directory"),
    sample_rate: int = typer.Option(DEFAULT_SAMPLE_RATE, help="Target sample rate"),
):
    input_path = Path(input_file)
    output_path = Path(output_dir)

    ensure_file_exists(input_path)
    ensure_supported_extension(input_path, SUPPORTED_EXTENSIONS)

    separator = DummySeparator()

    result = run_separation_pipeline(
        input_path=input_path,
        separator=separator,
        output_dir=output_path,
        sample_rate=sample_rate,
    )

    typer.echo("Separation finished.")
    typer.echo(f"Input: {result['input_file']}")
    typer.echo(f"Output dir: {result['output_dir']}")
    typer.echo(f"Sample rate: {result['sample_rate']}")
    typer.echo(f"Stems: {', '.join(result['stems'])}")
    typer.echo(f"Elapsed time: {result['elapsed_seconds']:.3f} s")


if __name__ == "__main__":
    app()