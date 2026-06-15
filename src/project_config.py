import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "experiment_config.json"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON experiment configuration.",
    )


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["_config_path"] = str(path)
    return config


def project_path(relative_path: str) -> Path:
    return (PROJECT_ROOT / relative_path).resolve()


def load_clean_dataset(config: dict):
    data_path = project_path(config["feature_file"])
    target_column = config["target_column"]
    df = pd.read_csv(data_path)

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)

    missing_count = int(df.isna().sum().sum())
    if missing_count:
        raise ValueError(f"Dataset contains missing values: {missing_count}")
    if target_column not in df.columns:
        raise ValueError(f"Dataset does not contain target column: {target_column}")

    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y, duplicate_count


def output_path(config: dict, filename: str) -> Path:
    directory = project_path(config["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def write_experiment_manifest(config: dict, stage: str) -> Path:
    config_path = Path(config["_config_path"])
    feature_path = project_path(config["feature_file"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None

    packages = {
        name: importlib.metadata.version(name)
        for name in ["librosa", "numpy", "pandas", "scikit-learn"]
    }
    source_hashes = {}
    for relative_path in [
        "src/project_config.py",
        "src/extract_features.py",
        "src/baseline_models.py",
        "src/target_model.py",
        "src/results_analysis.py",
    ]:
        source_path = project_path(relative_path)
        source_hashes[relative_path] = hashlib.sha256(source_path.read_bytes()).hexdigest()

    manifest_path = output_path(config, "experiment_manifest.json")
    manifest = {"runs": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest.setdefault("runs", []).append(
        {
            "stage": stage,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "dirty_worktree": dirty,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "feature_file_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
            "source_sha256": source_hashes,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
