import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from src.config import SUPPORTED_EXTENSIONS


EPSILON = 1e-12
MINIMUM_SI_SDR_DB = -120.0


class EvaluationError(ValueError):
    """Raised when estimates and references cannot be compared safely."""


def scale_invariant_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """Return global SI-SDR in dB, or None for a silent reference."""
    reference_vector = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate_vector = np.asarray(estimate, dtype=np.float64).reshape(-1)

    if reference_vector.shape != estimate_vector.shape:
        raise EvaluationError("Reference and estimate must have the same shape")

    sample_count = reference_vector.size
    reference_mean = float(np.mean(reference_vector))
    estimate_mean = float(np.mean(estimate_vector))
    reference_energy = max(
        float(np.dot(reference_vector, reference_vector))
        - sample_count * reference_mean * reference_mean,
        0.0,
    )
    estimate_energy = max(
        float(np.dot(estimate_vector, estimate_vector))
        - sample_count * estimate_mean * estimate_mean,
        0.0,
    )

    if reference_energy <= EPSILON:
        return None
    if estimate_energy <= EPSILON:
        return MINIMUM_SI_SDR_DB

    cross_energy = (
        float(np.dot(estimate_vector, reference_vector))
        - sample_count * estimate_mean * reference_mean
    )
    target_energy = cross_energy * cross_energy / reference_energy
    residual_energy = max(estimate_energy - target_energy, 0.0)
    return float(10.0 * np.log10((target_energy + EPSILON) / (residual_energy + EPSILON)))


def source_to_distortion_ratio(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float | None:
    """Return scale-dependent global SDR in dB, or None for a silent reference."""
    reference_vector = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate_vector = np.asarray(estimate, dtype=np.float64).reshape(-1)

    if reference_vector.shape != estimate_vector.shape:
        raise EvaluationError("Reference and estimate must have the same shape")

    reference_energy = float(np.dot(reference_vector, reference_vector))
    if reference_energy <= EPSILON:
        return None

    estimate_energy = float(np.dot(estimate_vector, estimate_vector))
    cross_energy = float(np.dot(reference_vector, estimate_vector))
    error_energy = max(reference_energy + estimate_energy - 2.0 * cross_energy, 0.0)
    return float(10.0 * np.log10((reference_energy + EPSILON) / (error_energy + EPSILON)))


def load_stem_mapping(path: Path) -> dict[str, list[str]]:
    """Load a reference-stem -> estimated-stems mapping from JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Stem mapping file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "stem_mapping" in payload:
        payload = payload["stem_mapping"]

    if not isinstance(payload, dict) or not payload:
        raise EvaluationError("Stem mapping must be a non-empty JSON object")

    mapping: dict[str, list[str]] = {}
    for reference_stem, estimated_stems in payload.items():
        if not isinstance(reference_stem, str) or not reference_stem.strip():
            raise EvaluationError("Every reference stem name must be a non-empty string")
        if isinstance(estimated_stems, str):
            estimated_stems = [estimated_stems]
        if not isinstance(estimated_stems, list) or not estimated_stems:
            raise EvaluationError(
                f"Mapping for '{reference_stem}' must contain at least one estimated stem"
            )
        if not all(isinstance(name, str) and name.strip() for name in estimated_stems):
            raise EvaluationError(
                f"Mapping for '{reference_stem}' contains an invalid stem name"
            )
        mapping[reference_stem.lower()] = [name.lower() for name in estimated_stems]

    return mapping


def discover_stems(directory: Path) -> dict[str, Path]:
    """Return audio files keyed by case-insensitive filename stem."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Stem directory does not exist: {directory}")

    stems: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stem_name = path.stem.lower()
        if stem_name in stems:
            raise EvaluationError(
                f"Duplicate stem name '{stem_name}' in directory: {directory}"
            )
        stems[stem_name] = path

    if not stems:
        raise EvaluationError(f"No supported audio stems found in: {directory}")
    return stems


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        str(path),
        dtype="float64",
        always_2d=True,
    )
    if audio.size == 0:
        raise EvaluationError(f"Audio file is empty: {path}")
    if not np.all(np.isfinite(audio)):
        raise EvaluationError(f"Audio file contains non-finite samples: {path}")
    return audio, int(sample_rate)


def _resample(audio: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return audio

    try:
        import librosa
    except ImportError as error:
        raise EvaluationError(
            "librosa is required to evaluate files with different sample rates"
        ) from error

    channels = [
        librosa.resample(
            audio[:, channel],
            orig_sr=original_rate,
            target_sr=target_rate,
        )
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1)


def _match_channels(reference: np.ndarray, audio: np.ndarray, label: str) -> np.ndarray:
    if reference.shape[1] == audio.shape[1]:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, reference.shape[1], axis=1)
    if reference.shape[1] == 1:
        return np.mean(audio, axis=1, keepdims=True)
    raise EvaluationError(
        f"Channel mismatch for {label}: reference has {reference.shape[1]}, "
        f"audio has {audio.shape[1]}"
    )


def _trim_to_reference(
    reference: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    label: str,
    max_length_mismatch_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    difference = abs(reference.shape[0] - audio.shape[0])
    maximum_difference = round(max_length_mismatch_seconds * sample_rate)
    if difference > maximum_difference:
        raise EvaluationError(
            f"Length mismatch for {label} is {difference / sample_rate:.3f} seconds; "
            f"maximum allowed is {max_length_mismatch_seconds:.3f} seconds"
        )

    sample_count = min(reference.shape[0], audio.shape[0])
    return reference[:sample_count], audio[:sample_count], difference


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(np.mean(usable))


def _median(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return float(np.median(usable))


def build_evaluation_history_row(
    evaluation: dict,
    evaluation_file: Path,
) -> dict:
    """Flatten an evaluation result into one comparison-friendly CSV row."""
    run = evaluation.get("run", {})
    summary = evaluation["summary"]
    return {
        "run_name": run.get("run_name", Path(evaluation["estimates_dir"]).parent.name),
        "model_name": run.get("model_name"),
        "device": run.get("device"),
        "segment": run.get("segment"),
        "shifts": run.get("shifts"),
        "mp3": run.get("mp3"),
        "input_file": run.get("input_file", evaluation.get("mixture_file")),
        "references_dir": evaluation["references_dir"],
        "stem_mapping": evaluation["stem_mapping"],
        "evaluated_stems": summary["evaluated_stems"],
        "evaluated_stem_count": summary["evaluated_stem_count"],
        "mean_si_sdr_db": summary["mean_si_sdr_db"],
        "median_si_sdr_db": summary["median_si_sdr_db"],
        "mean_sdr_db": summary["mean_sdr_db"],
        "median_sdr_db": summary["median_sdr_db"],
        "mean_si_sdri_db": summary["mean_si_sdri_db"],
        "median_si_sdri_db": summary["median_si_sdri_db"],
        "evaluation_file": str(evaluation_file),
        "created_at": evaluation["created_at"],
    }


def evaluate_stem_directories(
    estimates_dir: Path,
    references_dir: Path,
    mixture_file: Path | None = None,
    stem_mapping: dict[str, list[str]] | None = None,
    max_length_mismatch_seconds: float = 1.0,
) -> dict:
    """Evaluate matching stems and return JSON-serializable detailed results.

    ``stem_mapping`` maps each reference stem to one or more estimated stems.
    Multiple estimates are summed, which allows a six-stem result to be compared
    with four-stem references (for example, other = other + guitar + piano).
    """
    if max_length_mismatch_seconds < 0:
        raise EvaluationError("Maximum length mismatch cannot be negative")

    estimates = discover_stems(estimates_dir)
    references = discover_stems(references_dir)

    if stem_mapping is None:
        resolved_mapping = {stem: [stem] for stem in references}
    else:
        resolved_mapping = {
            reference.lower(): [estimate.lower() for estimate in estimate_names]
            for reference, estimate_names in stem_mapping.items()
        }

    mixture_audio: np.ndarray | None = None
    mixture_sample_rate: int | None = None
    if mixture_file is not None:
        if not mixture_file.is_file():
            raise FileNotFoundError(f"Mixture file does not exist: {mixture_file}")
        mixture_audio, mixture_sample_rate = _read_audio(mixture_file)

    per_stem: dict[str, dict] = {}
    missing_estimates: dict[str, list[str]] = {}
    missing_references = sorted(set(resolved_mapping) - set(references))
    used_estimates: set[str] = set()

    for reference_name, estimate_names in sorted(resolved_mapping.items()):
        if reference_name not in references:
            continue

        absent = [name for name in estimate_names if name not in estimates]
        if absent:
            missing_estimates[reference_name] = absent
            continue

        reference, sample_rate = _read_audio(references[reference_name])
        original_reference_samples = reference.shape[0]
        combined_estimate: np.ndarray | None = None
        component_details: list[dict] = []

        for estimate_name in estimate_names:
            estimate, estimate_rate = _read_audio(estimates[estimate_name])
            estimate = _resample(estimate, estimate_rate, sample_rate)
            estimate = _match_channels(reference, estimate, estimate_name)
            _, estimate, difference = _trim_to_reference(
                reference,
                estimate,
                sample_rate,
                estimate_name,
                max_length_mismatch_seconds,
            )

            if combined_estimate is None:
                combined_estimate = estimate
            else:
                shared_samples = min(combined_estimate.shape[0], estimate.shape[0])
                combined_estimate = (
                    combined_estimate[:shared_samples] + estimate[:shared_samples]
                )
            used_estimates.add(estimate_name)
            component_details.append(
                {
                    "stem": estimate_name,
                    "file": str(estimates[estimate_name]),
                    "original_sample_rate": estimate_rate,
                    "length_difference_samples": difference,
                }
            )

        if combined_estimate is None:
            continue

        evaluated_samples = min(reference.shape[0], combined_estimate.shape[0])
        reference = reference[:evaluated_samples]
        combined_estimate = combined_estimate[:evaluated_samples]

        si_sdr = scale_invariant_sdr(reference, combined_estimate)
        sdr = source_to_distortion_ratio(reference, combined_estimate)

        mixture_si_sdr: float | None = None
        si_sdri: float | None = None
        if mixture_audio is not None and mixture_sample_rate is not None:
            prepared_mixture = _resample(mixture_audio, mixture_sample_rate, sample_rate)
            prepared_mixture = _match_channels(reference, prepared_mixture, "mixture")
            baseline_reference, prepared_mixture, _ = _trim_to_reference(
                reference,
                prepared_mixture,
                sample_rate,
                "mixture",
                max_length_mismatch_seconds,
            )
            evaluation_samples = min(
                baseline_reference.shape[0],
                combined_estimate.shape[0],
            )
            baseline_reference = baseline_reference[:evaluation_samples]
            prepared_mixture = prepared_mixture[:evaluation_samples]
            estimate_for_improvement = combined_estimate[:evaluation_samples]
            estimate_si_sdr = scale_invariant_sdr(
                baseline_reference,
                estimate_for_improvement,
            )
            mixture_si_sdr = scale_invariant_sdr(
                baseline_reference,
                prepared_mixture,
            )
            if estimate_si_sdr is not None and mixture_si_sdr is not None:
                si_sdri = estimate_si_sdr - mixture_si_sdr

        per_stem[reference_name] = {
            "reference_file": str(references[reference_name]),
            "estimated_components": component_details,
            "sample_rate": sample_rate,
            "channels": reference.shape[1],
            "reference_samples": original_reference_samples,
            "evaluated_samples": reference.shape[0],
            "duration_seconds": reference.shape[0] / sample_rate,
            "metrics": {
                "si_sdr_db": si_sdr,
                "sdr_db": sdr,
                "mixture_si_sdr_db": mixture_si_sdr,
                "si_sdri_db": si_sdri,
            },
        }

    if not per_stem:
        raise EvaluationError(
            "No stems could be evaluated. Check filenames and the stem mapping."
        )

    si_sdr_values = [item["metrics"]["si_sdr_db"] for item in per_stem.values()]
    sdr_values = [item["metrics"]["sdr_db"] for item in per_stem.values()]
    si_sdri_values = [item["metrics"]["si_sdri_db"] for item in per_stem.values()]

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimates_dir": str(estimates_dir),
        "references_dir": str(references_dir),
        "mixture_file": str(mixture_file) if mixture_file is not None else None,
        "stem_mapping": resolved_mapping,
        "summary": {
            "reference_stem_count": len(references),
            "evaluated_stem_count": len(per_stem),
            "evaluated_stems": sorted(per_stem),
            "missing_estimates": missing_estimates,
            "missing_references": missing_references,
            "unused_estimates": sorted(set(estimates) - used_estimates),
            "mean_si_sdr_db": _mean(si_sdr_values),
            "median_si_sdr_db": _median(si_sdr_values),
            "mean_sdr_db": _mean(sdr_values),
            "median_sdr_db": _median(sdr_values),
            "mean_si_sdri_db": _mean(si_sdri_values),
            "median_si_sdri_db": _median(si_sdri_values),
        },
        "per_stem": per_stem,
    }
