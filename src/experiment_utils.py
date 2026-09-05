import csv
import json
from datetime import datetime
from pathlib import Path


def generate_run_name(input_path: Path, custom_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = custom_name if custom_name else input_path.stem
    return f"run_{timestamp}_{base_name}"


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def append_experiment_to_csv(data: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_data = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in data.items()
    }

    existing_rows: list[dict] = []
    existing_fieldnames: list[str] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fieldnames = existing_fieldnames + [
        key for key in serialized_data if key not in existing_fieldnames
    ]

    if not existing_fieldnames:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(serialized_data)
        return

    if fieldnames != existing_fieldnames:
        temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with open(temporary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(serialized_data)
        temporary_path.replace(csv_path)
        return

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(serialized_data)
