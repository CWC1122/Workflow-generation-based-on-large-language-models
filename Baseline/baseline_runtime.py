from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mrg_sc_runtime import (
    SUPPORTED_DATASETS,
    build_data_path,
    get_dataset_name,
    get_embedding_model,
    get_embedding_timeout,
    get_embedding_url,
    get_split_dir,
)


BASELINE_ARTIFACTS_ROOT = Path(
    os.getenv("MRG_SC_BASELINE_ARTIFACTS_ROOT", str(REPO_ROOT / "Baseline" / "artifacts"))
).expanduser().resolve()


def train_data_dir(dataset: str | None = None) -> str:
    return str(get_split_dir("train", dataset))


def test_data_dir(dataset: str | None = None) -> str:
    return str(get_split_dir("test", dataset))


def baseline_train_path(filename: str, dataset: str | None = None) -> str:
    return build_data_path("train", filename, dataset)


def baseline_test_path(filename: str, dataset: str | None = None) -> str:
    return build_data_path("test", filename, dataset)


def baseline_artifact_path(baseline_name: str, filename: str, dataset: str | None = None) -> str:
    dataset_name = get_dataset_name(dataset)
    return str(BASELINE_ARTIFACTS_ROOT / baseline_name / dataset_name / filename)


def get_word2vec_binary_path() -> str:
    default_path = REPO_ROOT / "models" / "GoogleNews-vectors-negative300.bin"
    return os.getenv("MRG_SC_WORD2VEC_PATH", str(default_path)).strip()


def get_baseline_llm_model(default_model: str) -> str:
    return os.getenv("MRG_SC_LLM_MODEL", default_model).strip()


def get_baseline_llm_base_url(default_base_url: str = "http://localhost:11434/v1/") -> str:
    return os.getenv("MRG_SC_LLM_BASE_URL", default_base_url).strip()


def get_baseline_embedding_url(default_url: str | None = None) -> str:
    configured = os.getenv("MRG_SC_EMBEDDING_URL")
    if configured:
        return configured.strip()
    if default_url is not None:
        return default_url.strip()
    return get_embedding_url()


def get_baseline_embedding_model(default_model: str | None = None) -> str:
    configured = os.getenv("MRG_SC_EMBEDDING_MODEL")
    if configured:
        return configured.strip()
    if default_model is not None:
        return default_model.strip()
    return get_embedding_model()


def get_baseline_embedding_timeout() -> float:
    return float(get_embedding_timeout())


def get_baseline_dataset_list(default: list[str] | None = None) -> list[str]:
    raw = os.getenv("MRG_SC_DATASETS", "").strip()
    if not raw:
        if default is not None:
            return default
        return [get_dataset_name()]

    datasets = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{item}'. Expected one of: {', '.join(SUPPORTED_DATASETS)}."
            )
        datasets.append(name)

    if not datasets:
        raise ValueError("MRG_SC_DATASETS was provided but no valid dataset names were found.")
    return datasets
