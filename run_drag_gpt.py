import argparse
import gc
import importlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
DRAG_DIR = THIS_DIR / "DRAG"


# ==================== User Config ====================
# Data layout:
#   DATA_ROOT_DIR/train/<dataset>/{data.json, tool_desc.json, graph_desc.json}
#   DATA_ROOT_DIR/test/<dataset>/{data.json, tool_desc.json, graph_desc.json}
# If train/ or test/ does not exist, DATA_ROOT_DIR/<dataset> is used instead.
DATA_ROOT_DIR = str(THIS_DIR / "Data")
HISTORY_ROOT_DIR = None
TEST_ROOT_DIR = None

# Output layout:
#   OUTPUT_ROOT_DIR/<model_name>/<dataset>/stage*.json
OUTPUT_ROOT_DIR = str(THIS_DIR / "outputs" / "drag_gpt")

DATASETS_TO_RUN = ["daily", "hug", "mul", "ultratool"]
TASK_LIMIT = None
ENABLE_RESUME = True

# GPT-compatible endpoint for Stage3 DAG planning and Stage6 path selection.
GPT_MODEL = "gpt-4o-mini"
GPT_DAG_MODEL = None
GPT_PATH_MODEL = None
GPT_BASE_URL = "https://api.openai.com/v1"
GPT_API_STYLE = "chat"  # "chat" or "responses"
GPT_API_KEY = ""  # Prefer setting OPENAI_API_KEY instead of hard-coding keys.
GPT_API_KEYS = []  # Optional: ["sk-...", "sk-..."] for parallel calls.
GPT_PROXY_URL = None

# Embedding service used by retrieval stages. The stage modules also read
# DRAG_EMBEDDING_URL from the environment.
EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"

STAGE1_TOP_K = 4
STAGE5_CANDIDATE_TOPK = 4
BEAM_WIDTH = 3
STAGE3_WORKERS = 1
STAGE6_WORKERS = 1
SAVE_EVERY_N = 20
PRINT_EVERY_N = 20
VERBOSE_PROGRESS = True
# ==================== User Config Ends ====================


STAGE_LABELS = {
    "stage1": "Stage1",
    "stage2": "Stage2",
    "stage3": "Stage3",
    "stage4": "Stage4",
    "stage5": "Stage5",
    "stage6": "Stage6",
    "stage7": "Stage7",
}

FULL_FLAGS = {
    "stage1_top_k": STAGE1_TOP_K,
    "use_adaptive_preselection": True,
    "fixed_stage2_k": 16,
    "stage2_top_task_k": None,
    "stage5_candidate_topk": STAGE5_CANDIDATE_TOPK,
    "use_score_filter": True,
    "use_io_filter": True,
    "use_graph_filter": True,
    "use_strong_filter": True,
    "beam_width": BEAM_WIDTH,
    "stage3_workers": STAGE3_WORKERS,
    "stage6_workers": STAGE6_WORKERS,
}

_LLM_THREAD_STATE = threading.local()
_ACTIVE_LLM_API_KEYS: List[str] = []


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as file_obj:
        return json.load(file_obj)


def save_json(path: str, data: Any) -> None:
    parent_dir = os.path.dirname(path) or "."
    os.makedirs(parent_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_json_", suffix=".json", dir=parent_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return value.strip("_") or "item"


def _sort_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(tasks, key=lambda item: str(item.get("annotation_id", "")))


def _get_task_id(task: Dict[str, Any]) -> str:
    return str(task.get("annotation_id", ""))


def _is_nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _is_valid_stage_output(stage_name: str, task: Dict[str, Any]) -> bool:
    if stage_name == "Stage1":
        try:
            return int(task.get("pred_task_num", 0)) > 0
        except Exception:
            return False
    if stage_name == "Stage2":
        return _is_nonempty_list(task.get("st1.5_service"))
    if stage_name == "Stage3":
        return _is_nonempty_list(task.get("nodes"))
    if stage_name == "Stage4":
        return _is_nonempty_list(task.get("chains"))
    if stage_name == "Stage5":
        return _is_nonempty_list(task.get("chain_candidates"))
    if stage_name == "Stage6":
        return _is_nonempty_list(task.get("chain_selected_services"))
    if stage_name == "Stage7":
        return _is_nonempty_list(task.get("recom_result"))
    return True


def _save_stage_output(output_path: Path, tasks: List[Dict[str, Any]]) -> None:
    save_json(str(output_path), _sort_tasks(tasks))


def _unfinished_path_for_stage(output_path: Path, stage_name: str) -> Optional[Path]:
    if stage_name == "Stage3":
        return output_path.with_name("unfinished_stage3.json")
    if stage_name == "Stage6":
        return output_path.with_name("unfinished_stage6.json")
    return None


def _load_unfinished_records_for_stage(output_path: Path, stage_name: str) -> List[Dict[str, Any]]:
    unfinished_path = _unfinished_path_for_stage(output_path, stage_name)
    if unfinished_path is None or not unfinished_path.exists():
        return []
    try:
        records = load_json(str(unfinished_path))
    except Exception:
        return []
    return [record for record in records if isinstance(record, dict)]


def _load_unfinished_ids_for_stage(output_path: Path, stage_name: str) -> set:
    return {_get_task_id(record) for record in _load_unfinished_records_for_stage(output_path, stage_name)}


def _prepare_stage_resume(
    input_tasks: List[Dict[str, Any]],
    output_path: Path,
    resume: bool,
    stage_name: str,
):
    if not resume or not output_path.exists():
        skipped_ids = _load_unfinished_ids_for_stage(output_path, stage_name) if resume else set()
        remaining = [task for task in input_tasks if _get_task_id(task) not in skipped_ids]
        return [], skipped_ids, remaining, 0

    try:
        existing = load_json(str(output_path))
    except Exception as exc:
        if VERBOSE_PROGRESS:
            print(f"[resume] {stage_name}: failed to read {output_path.name}, rerun all ({exc})")
        skipped_ids = _load_unfinished_ids_for_stage(output_path, stage_name)
        remaining = [task for task in input_tasks if _get_task_id(task) not in skipped_ids]
        return [], skipped_ids, remaining, 0

    if not isinstance(existing, list):
        skipped_ids = _load_unfinished_ids_for_stage(output_path, stage_name)
        remaining = [task for task in input_tasks if _get_task_id(task) not in skipped_ids]
        return [], skipped_ids, remaining, 0

    input_ids = {_get_task_id(task) for task in input_tasks}
    skipped_ids = _load_unfinished_ids_for_stage(output_path, stage_name) & input_ids
    processed = []
    invalid_count = 0
    for task in existing:
        if _get_task_id(task) not in input_ids:
            continue
        if _is_valid_stage_output(stage_name, task):
            processed.append(task)
        else:
            invalid_count += 1
    processed_ids = {_get_task_id(task) for task in processed} | skipped_ids
    remaining = [task for task in input_tasks if _get_task_id(task) not in processed_ids]
    return _sort_tasks(processed), processed_ids, remaining, invalid_count


def _checkpoint_stage_output(
    dataset: str,
    stage_name: str,
    output_path: Path,
    processed: List[Dict[str, Any]],
    total: int,
    force: bool = False,
) -> None:
    if not force and len(processed) % SAVE_EVERY_N != 0:
        return
    _save_stage_output(output_path, processed)
    if VERBOSE_PROGRESS:
        print(f"[{dataset}] {stage_name}: saved {len(processed)}/{total} -> {output_path.name}")


def _prime_stage_output_file(
    dataset: str,
    stage_name: str,
    output_path: Path,
    processed: List[Dict[str, Any]],
    invalid_count: int,
) -> None:
    if invalid_count > 0 or not output_path.exists():
        _save_stage_output(output_path, processed)
        if VERBOSE_PROGRESS:
            print(f"[{dataset}] {stage_name}: primed {output_path.name} with {len(processed)} cached items")


def _log_resume_state(
    dataset: str,
    stage_name: str,
    output_path: Path,
    processed_count: int,
    total_count: int,
    invalid_count: int = 0,
) -> None:
    if not VERBOSE_PROGRESS:
        return
    if processed_count <= 0:
        message = f"[{dataset}] {stage_name}: start fresh"
    elif processed_count >= total_count:
        message = f"[{dataset}] {stage_name}: resume hit, skip {processed_count}/{total_count}"
    else:
        message = f"[{dataset}] {stage_name}: resume from {processed_count}/{total_count}"
    if invalid_count > 0:
        message += f"; discarded invalid cached items={invalid_count}"
    print(f"{message} ({output_path.name})")


def _normalize_workers(value: Optional[int]) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _activate_stage_dir(stage_dir: Path, clear_modules: Optional[List[str]] = None) -> None:
    for module_name in clear_modules or []:
        if module_name in sys.modules:
            del sys.modules[module_name]

    stage_dir_str = str(stage_dir)
    if stage_dir_str in sys.path:
        sys.path.remove(stage_dir_str)
    sys.path.insert(0, stage_dir_str)
    importlib.invalidate_caches()


def _import_stage_module(
    alias: str,
    stage_dir: Path,
    module_name: str,
    clear_modules: Optional[List[str]] = None,
):
    _activate_stage_dir(stage_dir, clear_modules=clear_modules)
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    sys.modules[alias] = module
    return module


def _load_stage_modules() -> Dict[str, Any]:
    modules = {}

    stage1_dir = DRAG_DIR / "Stage1_TaskNumPrediction_train"
    modules["stage1_embedding"] = _import_stage_module(
        "drag_stage1_embedding",
        stage1_dir,
        "embedding_client",
        clear_modules=["embedding_client", "embedding_cache", "file_utils", "task_retriever"],
    )
    modules["stage1_cache"] = _import_stage_module(
        "drag_stage1_cache",
        stage1_dir,
        "embedding_cache",
        clear_modules=["embedding_client", "embedding_cache", "file_utils", "task_retriever"],
    )
    modules["stage1_retriever"] = _import_stage_module(
        "drag_stage1_retriever",
        stage1_dir,
        "task_retriever",
        clear_modules=["embedding_client", "embedding_cache", "file_utils", "task_retriever"],
    )

    stage2_dir = DRAG_DIR / "Stage2_ServicePreselect_train"
    modules["stage2_embedding"] = _import_stage_module(
        "drag_stage2_embedding",
        stage2_dir,
        "embedding_client",
        clear_modules=["embedding_client", "embedding_cache", "file_utils", "train_service_retriever"],
    )
    modules["stage2_cache"] = _import_stage_module(
        "drag_stage2_cache",
        stage2_dir,
        "embedding_cache",
        clear_modules=["embedding_client", "embedding_cache", "file_utils", "train_service_retriever"],
    )
    modules["stage2_retriever"] = _import_stage_module(
        "drag_stage2_retriever",
        stage2_dir,
        "train_service_retriever",
        clear_modules=["embedding_client", "embedding_cache", "file_utils", "train_service_retriever"],
    )

    stage3_dir = DRAG_DIR / "Stage3_TaskDAG"
    modules["stage3"] = _import_stage_module(
        "drag_stage3",
        stage3_dir,
        "dag_generator",
        clear_modules=["utils", "file_utils", "dag_generator"],
    )

    stage4_dir = DRAG_DIR / "Stage4_ChainBuild"
    modules["stage4_embedding"] = _import_stage_module(
        "drag_stage4_embedding",
        stage4_dir,
        "embedding_client",
        clear_modules=["embedding_client", "utils", "chain_builder"],
    )
    modules["stage4_builder"] = _import_stage_module(
        "drag_stage4_builder",
        stage4_dir,
        "chain_builder",
        clear_modules=["embedding_client", "utils", "chain_builder"],
    )

    stage5_dir = DRAG_DIR / "Stage5_ServiceRecall_Strict"
    modules["stage5_embedding"] = _import_stage_module(
        "drag_stage5_embedding",
        stage5_dir,
        "embedding_client",
        clear_modules=["embedding_client", "utils", "recall_filter"],
    )
    modules["stage5_filter"] = _import_stage_module(
        "drag_stage5_filter",
        stage5_dir,
        "recall_filter",
        clear_modules=["embedding_client", "utils", "recall_filter"],
    )

    stage6_dir = DRAG_DIR / "Stage6"
    modules["stage6_embedding"] = _import_stage_module(
        "drag_stage6_embedding",
        stage6_dir,
        "embedding_client",
        clear_modules=["embedding_client", "utils", "service_selector"],
    )
    modules["stage6_selector"] = _import_stage_module(
        "drag_stage6_selector",
        stage6_dir,
        "service_selector",
        clear_modules=["embedding_client", "utils", "service_selector"],
    )

    stage7_dir = DRAG_DIR / "Stage7"
    modules["stage7_merger"] = _import_stage_module(
        "drag_stage7_merger",
        stage7_dir,
        "chain_merger",
        clear_modules=["chain_merger"],
    )

    gpt_dir = DRAG_DIR / "gptLLM"
    modules["gpt_llm"] = _import_stage_module(
        "drag_gpt_llm",
        gpt_dir,
        "gptLLM",
        clear_modules=["gptLLM"],
    )

    return modules


def _split_api_keys(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [key.strip() for key in value.split(",") if key.strip()]


def _set_active_llm_api_keys(api_keys: List[str]) -> None:
    global _ACTIVE_LLM_API_KEYS
    _ACTIVE_LLM_API_KEYS = [key.strip() for key in api_keys if key and key.strip()]


def _active_llm_api_keys() -> List[str]:
    return _ACTIVE_LLM_API_KEYS


def _llm_api_key_for_index(index: int) -> str:
    keys = _active_llm_api_keys()
    if not keys:
        raise ValueError("No GPT API key configured. Set OPENAI_API_KEY or OPENAI_API_KEYS.")
    return keys[index % len(keys)]


def _current_thread_llm_api_key(default_api_key: str) -> str:
    return getattr(_LLM_THREAD_STATE, "api_key", None) or default_api_key


def _run_with_thread_llm_api_key(api_key: str, fn, *args, **kwargs):
    previous = getattr(_LLM_THREAD_STATE, "api_key", None)
    _LLM_THREAD_STATE.api_key = api_key
    try:
        return fn(*args, **kwargs)
    finally:
        if previous is None:
            try:
                delattr(_LLM_THREAD_STATE, "api_key")
            except AttributeError:
                pass
        else:
            _LLM_THREAD_STATE.api_key = previous


def _patch_gpt_runtime(
    modules: Dict[str, Any],
    dag_model: str,
    path_model: str,
    base_url: str,
    api_key: str,
    api_style: str,
    proxy_url: Optional[str],
) -> None:
    gpt_call = modules["gpt_llm"].gptLLM

    def _stage3_gpt(messages, model=None, temperature=0.0, **kwargs):
        return gpt_call(
            messages,
            model=dag_model,
            temperature=temperature,
            base_url=base_url,
            api_key=_current_thread_llm_api_key(api_key),
            api_style=api_style,
            proxy_url=proxy_url,
            **kwargs,
        )

    def _stage6_gpt(messages, model=None, temperature=0.0, **kwargs):
        return gpt_call(
            messages,
            model=path_model,
            temperature=temperature,
            base_url=base_url,
            api_key=_current_thread_llm_api_key(api_key),
            api_style=api_style,
            proxy_url=proxy_url,
            **kwargs,
        )

    modules["stage3"].localLLM = _stage3_gpt
    modules["stage6_selector"].localLLM = _stage6_gpt


def _build_structured_chains(chains, nodes):
    node_map = {node["id"]: node for node in nodes if "id" in node}
    structured = []
    for chain in chains:
        current_chain = []
        for index, node_id in enumerate(chain):
            node = node_map.get(node_id)
            if not node:
                continue
            text = node.get("requirement") or node.get("task") or ""
            if not text.strip():
                continue
            current_chain.append({"id": node_id, "requirement": text, "index": index})
        if current_chain:
            structured.append(current_chain)
    return structured


def _make_stage1_fallback_length(train_tasks: List[Dict[str, Any]]) -> int:
    lengths = [len(task.get("action_id_list", [])) for task in train_tasks if isinstance(task.get("action_id_list"), list)]
    lengths = [value for value in lengths if value > 0]
    if not lengths:
        return 3
    return max(1, int(round(float(np.mean(lengths)))))


def _run_stage1(
    dataset: str,
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    flags: Dict[str, Any],
    resume: bool,
    limit: Optional[int],
):
    test_tasks = load_json(str(test_dir / "data.json"))
    if limit is not None:
        test_tasks = test_tasks[:limit]
    if not test_tasks:
        _save_stage_output(output_path, [])
        return []

    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(test_tasks, output_path, resume, "Stage1")
    _log_resume_state(dataset, "Stage1", output_path, len(processed), len(test_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage1", output_path, processed, invalid_count)
    if not remaining:
        return processed

    train_tasks = load_json(str(train_dir / "data.json"))
    embedder = modules["stage1_embedding"].EmbeddingClient()
    embeddings, valid_train_tasks = modules["stage1_cache"].build_or_load_embeddings(
        train_tasks,
        str(train_dir / "stage1_history_embeddings.npy"),
    )
    retriever = modules["stage1_retriever"].TaskRetriever(valid_train_tasks, embeddings, have_self=True)
    fallback_length = _make_stage1_fallback_length(train_tasks)
    top_k = int(flags.get("stage1_top_k", STAGE1_TOP_K))

    bar = tqdm(remaining, desc=f"[{dataset}] Stage1", ncols=100, initial=len(processed), total=len(test_tasks))
    for task in bar:
        if _get_task_id(task) in processed_ids:
            continue
        item = deepcopy(task)
        try:
            query_embedding = embedder.get_embedding(item.get("confirmed_task", ""))
        except Exception as exc:
            print(f"[Stage1] embedding error: {exc}")
            query_embedding = None

        item["pred_task_num"] = retriever.predict_task_num(
            query_embedding,
            recom_num=item.get("recom_num", fallback_length),
            top_k=top_k,
        )
        processed.append(item)
        processed_ids.add(_get_task_id(item))
        if VERBOSE_PROGRESS and len(processed) % PRINT_EVERY_N == 0:
            bar.set_postfix_str(f"done={len(processed)}/{len(test_tasks)}")
        _checkpoint_stage_output(dataset, "Stage1", output_path, processed, len(test_tasks))

    _checkpoint_stage_output(dataset, "Stage1", output_path, processed, len(test_tasks), force=True)
    return processed


def _run_stage2(
    dataset: str,
    stage1_tasks: List[Dict[str, Any]],
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    flags: Dict[str, Any],
    resume: bool,
):
    if not stage1_tasks:
        _save_stage_output(output_path, [])
        return []

    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(stage1_tasks, output_path, resume, "Stage2")
    _log_resume_state(dataset, "Stage2", output_path, len(processed), len(stage1_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage2", output_path, processed, invalid_count)
    if not remaining:
        return processed

    train_tasks = load_json(str(train_dir / "data.json"))
    services = load_json(str(test_dir / "tool_desc.json"))
    embedder = modules["stage2_embedding"].EmbeddingClient()
    embeddings, valid_train_tasks = modules["stage2_cache"].build_or_load_embeddings(
        train_tasks,
        str(train_dir / "stage2_train_task_embeddings.npy"),
    )
    valid_service_ids = [service["action_uid"] for service in services]
    recall_engine = modules["stage2_retriever"].TrainTaskServiceRetriever(
        valid_train_tasks,
        embeddings,
        valid_service_ids,
    )

    bar = tqdm(remaining, desc=f"[{dataset}] Stage2", ncols=100, initial=len(processed), total=len(stage1_tasks))
    for task in bar:
        if _get_task_id(task) in processed_ids:
            continue
        item = deepcopy(task)
        task_num = int(item.get("pred_task_num", 3))
        candidate_size = min(max(12, task_num * 4), 24)

        try:
            query_embedding = embedder.get_embedding(item.get("confirmed_task", ""))
        except Exception as exc:
            print(f"[Stage2] embedding error: {exc}")
            query_embedding = None

        top_services = recall_engine.recall(query_embedding, candidate_size)
        item["st1.5_service"] = [service["action_uid"] for service in top_services]
        processed.append(item)
        processed_ids.add(_get_task_id(item))
        if VERBOSE_PROGRESS and len(processed) % PRINT_EVERY_N == 0:
            bar.set_postfix_str(f"done={len(processed)}/{len(stage1_tasks)}")
        _checkpoint_stage_output(dataset, "Stage2", output_path, processed, len(stage1_tasks))

    _checkpoint_stage_output(dataset, "Stage2", output_path, processed, len(stage1_tasks), force=True)
    return processed


def _run_stage3(
    dataset: str,
    stage2_tasks: List[Dict[str, Any]],
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    flags: Dict[str, Any],
    resume: bool,
):
    if not stage2_tasks:
        _save_stage_output(output_path, [])
        return []

    generator = modules["stage3"].DAGGenerator(str(test_dir / "tool_desc.json"))
    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(stage2_tasks, output_path, resume, "Stage3")
    _log_resume_state(dataset, "Stage3", output_path, len(processed), len(stage2_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage3", output_path, processed, invalid_count)
    if not remaining:
        return processed

    max_workers = min(_normalize_workers(flags.get("stage3_workers", STAGE3_WORKERS)), len(_active_llm_api_keys()))
    unfinished = _load_unfinished_records_for_stage(output_path, "Stage3") if resume else []
    bar = tqdm(remaining, desc=f"[{dataset}] Stage3", ncols=100, initial=len(processed_ids), total=len(stage2_tasks))

    def _process(task):
        item = deepcopy(task)
        error_reason = None
        try:
            dag = generator.generate_dag(
                item.get("confirmed_task", ""),
                int(item.get("pred_task_num", 3)),
                item.get("st1.5_service", []),
            )
            if dag is None:
                item["nodes"] = []
                item["edges"] = []
                error_reason = "generator_returned_none"
            else:
                item["nodes"] = dag.get("nodes", [])
                item["edges"] = dag.get("edges", [])
        except Exception as exc:
            item["nodes"] = []
            item["edges"] = []
            error_reason = str(exc)
        return item, error_reason

    if max_workers <= 1:
        for index, task in enumerate(bar):
            if _get_task_id(task) in processed_ids:
                continue
            item, error_reason = _run_with_thread_llm_api_key(_llm_api_key_for_index(index), _process, task)
            if error_reason:
                unfinished.append({"annotation_id": item.get("annotation_id"), "reason": error_reason})
                save_json(str(output_path.with_name("unfinished_stage3.json")), unfinished)
            else:
                processed.append(item)
                processed_ids.add(_get_task_id(item))
            _checkpoint_stage_output(dataset, "Stage3", output_path, processed, len(stage2_tasks), force=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_with_thread_llm_api_key, _llm_api_key_for_index(index), _process, task): task
                for index, task in enumerate(remaining)
            }
            for future in as_completed(futures):
                item, error_reason = future.result()
                if error_reason:
                    unfinished.append({"annotation_id": item.get("annotation_id"), "reason": error_reason})
                    save_json(str(output_path.with_name("unfinished_stage3.json")), unfinished)
                else:
                    processed.append(item)
                    processed_ids.add(_get_task_id(item))
                bar.update(1)
                _checkpoint_stage_output(dataset, "Stage3", output_path, processed, len(stage2_tasks), force=True)

    _checkpoint_stage_output(dataset, "Stage3", output_path, processed, len(stage2_tasks), force=True)
    if unfinished:
        save_json(str(output_path.with_name("unfinished_stage3.json")), unfinished)
    return processed


def _run_stage4(
    dataset: str,
    stage3_tasks: List[Dict[str, Any]],
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    resume: bool,
):
    if not stage3_tasks:
        _save_stage_output(output_path, [])
        return []

    services = load_json(str(test_dir / "tool_desc.json"))
    service_dict = {service["action_uid"]: service for service in services}
    embedder = modules["stage4_embedding"].EmbeddingClient()
    builder = modules["stage4_builder"].SemanticChainBuilder(embedder, service_dict)

    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(stage3_tasks, output_path, resume, "Stage4")
    _log_resume_state(dataset, "Stage4", output_path, len(processed), len(stage3_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage4", output_path, processed, invalid_count)
    if not remaining:
        return processed

    bar = tqdm(remaining, desc=f"[{dataset}] Stage4", ncols=100, initial=len(processed), total=len(stage3_tasks))
    for task in bar:
        if _get_task_id(task) in processed_ids:
            continue
        item = deepcopy(task)
        nodes = item.get("nodes", [])
        edges = item.get("edges", [])
        if not nodes:
            item["chains"] = []
        else:
            weights = builder.compute_node_weights(
                nodes,
                edges,
                item.get("confirmed_task", ""),
                item.get("st1.5_service", []),
            )
            chains = builder.build_chains(nodes, edges, weights)
            chains = builder.sort_chains(chains, weights)
            item["chains"] = _build_structured_chains(chains, nodes)
        processed.append(item)
        processed_ids.add(_get_task_id(item))
        _checkpoint_stage_output(dataset, "Stage4", output_path, processed, len(stage3_tasks))

    _checkpoint_stage_output(dataset, "Stage4", output_path, processed, len(stage3_tasks), force=True)
    return processed


def _run_stage5(
    dataset: str,
    stage4_tasks: List[Dict[str, Any]],
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    flags: Dict[str, Any],
    resume: bool,
):
    if not stage4_tasks:
        _save_stage_output(output_path, [])
        return []

    services = load_json(str(test_dir / "tool_desc.json"))
    graph_data = load_json(str(test_dir / "graph_desc.json"))
    embedder = modules["stage5_embedding"].EmbeddingClient()
    filter_engine = modules["stage5_filter"].RecallFilter(embedder, services, graph_data)
    candidate_topk = int(flags.get("stage5_candidate_topk", STAGE5_CANDIDATE_TOPK))

    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(stage4_tasks, output_path, resume, "Stage5")
    _log_resume_state(dataset, "Stage5", output_path, len(processed), len(stage4_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage5", output_path, processed, invalid_count)
    if not remaining:
        return processed

    bar = tqdm(remaining, desc=f"[{dataset}] Stage5", ncols=100, initial=len(processed), total=len(stage4_tasks))
    for task in bar:
        if _get_task_id(task) in processed_ids:
            continue
        item = deepcopy(task)
        chains_candidates = []
        st15 = set(item.get("st1.5_service", []))
        edges = item.get("edges", [])

        for chain in item.get("chains", []):
            node_list = []
            chain_len = len(chain)
            for idx, node in enumerate(chain):
                node_id = node["id"]
                text = node["requirement"]
                prev_node_id = chain[idx - 1]["id"] if idx > 0 else None
                next_node_id = chain[idx + 1]["id"] if idx < chain_len - 1 else None

                candidates = filter_engine.recall(text, topk=candidate_topk)
                fallback_candidate = candidates[0] if candidates else None
                if flags.get("use_score_filter", True):
                    candidates = filter_engine.filter_by_score(candidates)
                if flags.get("use_io_filter", True):
                    candidates = filter_engine.filter_by_io(node_id, prev_node_id, next_node_id, edges, candidates)
                if flags.get("use_strong_filter", True):
                    candidates = filter_engine.strong_filter(candidates, st15)
                if not candidates and fallback_candidate:
                    candidates = [fallback_candidate]

                node_list.append(
                    {
                        "id": node_id,
                        "requirement": text,
                        "candidates": candidates,
                        "fallback_candidate": fallback_candidate,
                    }
                )
            chains_candidates.append(node_list)

        if flags.get("use_graph_filter", True):
            chains_candidates = filter_engine.filter_by_graph(chains_candidates)
        item["chain_candidates"] = chains_candidates

        processed.append(item)
        processed_ids.add(_get_task_id(item))
        _checkpoint_stage_output(dataset, "Stage5", output_path, processed, len(stage4_tasks))

    _checkpoint_stage_output(dataset, "Stage5", output_path, processed, len(stage4_tasks), force=True)
    return processed


def _run_stage6(
    dataset: str,
    stage5_tasks: List[Dict[str, Any]],
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    flags: Dict[str, Any],
    resume: bool,
):
    if not stage5_tasks:
        _save_stage_output(output_path, [])
        return []

    services = load_json(str(test_dir / "tool_desc.json"))
    graph_data = load_json(str(test_dir / "graph_desc.json"))
    service_dict = {service["action_uid"]: service for service in services}
    graph = {}
    for link in graph_data:
        graph.setdefault(link["source"], set()).add(link["target"])

    embedder = modules["stage6_embedding"].EmbeddingClient()
    selector = modules["stage6_selector"].ServiceSelector(
        service_dict,
        graph,
        embedder,
        beam_width=int(flags.get("beam_width", BEAM_WIDTH)),
    )

    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(stage5_tasks, output_path, resume, "Stage6")
    _log_resume_state(dataset, "Stage6", output_path, len(processed), len(stage5_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage6", output_path, processed, invalid_count)
    if not remaining:
        return processed

    max_workers = min(_normalize_workers(flags.get("stage6_workers", STAGE6_WORKERS)), len(_active_llm_api_keys()))
    unfinished = _load_unfinished_records_for_stage(output_path, "Stage6") if resume else []
    bar = tqdm(remaining, desc=f"[{dataset}] Stage6", ncols=100, initial=len(processed_ids), total=len(stage5_tasks))

    def _process(task):
        item = deepcopy(task)
        try:
            return selector.select_for_task(item), None
        except Exception as exc:
            item["chain_selected_services"] = []
            item["stage6_error"] = str(exc)
            return item, str(exc)

    if max_workers <= 1:
        for index, task in enumerate(bar):
            if _get_task_id(task) in processed_ids:
                continue
            result, error_reason = _run_with_thread_llm_api_key(_llm_api_key_for_index(index), _process, task)
            if error_reason:
                unfinished.append({"annotation_id": result.get("annotation_id"), "reason": error_reason})
                save_json(str(output_path.with_name("unfinished_stage6.json")), unfinished)
            else:
                processed.append(result)
                processed_ids.add(_get_task_id(result))
            _checkpoint_stage_output(dataset, "Stage6", output_path, processed, len(stage5_tasks), force=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_with_thread_llm_api_key, _llm_api_key_for_index(index), _process, task): task
                for index, task in enumerate(remaining)
            }
            for future in as_completed(futures):
                result, error_reason = future.result()
                if error_reason:
                    unfinished.append({"annotation_id": result.get("annotation_id"), "reason": error_reason})
                    save_json(str(output_path.with_name("unfinished_stage6.json")), unfinished)
                else:
                    processed.append(result)
                    processed_ids.add(_get_task_id(result))
                bar.update(1)
                _checkpoint_stage_output(dataset, "Stage6", output_path, processed, len(stage5_tasks), force=True)

    _checkpoint_stage_output(dataset, "Stage6", output_path, processed, len(stage5_tasks), force=True)
    if unfinished:
        save_json(str(output_path.with_name("unfinished_stage6.json")), unfinished)
    return processed


def _run_stage7(
    dataset: str,
    stage6_tasks: List[Dict[str, Any]],
    test_dir: Path,
    output_path: Path,
    modules: Dict[str, Any],
    resume: bool,
):
    if not stage6_tasks:
        _save_stage_output(output_path, [])
        return []

    services = load_json(str(test_dir / "tool_desc.json"))
    service_dict = {service["action_uid"]: service for service in services}
    chain_merger_cls = modules["stage7_merger"].ChainMerger

    processed, processed_ids, remaining, invalid_count = _prepare_stage_resume(stage6_tasks, output_path, resume, "Stage7")
    _log_resume_state(dataset, "Stage7", output_path, len(processed), len(stage6_tasks), invalid_count)
    _prime_stage_output_file(dataset, "Stage7", output_path, processed, invalid_count)
    if not remaining:
        return processed

    unfinished = []
    bar = tqdm(remaining, desc=f"[{dataset}] Stage7", ncols=100, initial=len(processed), total=len(stage6_tasks))
    for task in bar:
        if _get_task_id(task) in processed_ids:
            continue
        item = deepcopy(task)
        try:
            merger = chain_merger_cls(item.get("edges", []), service_dict)
            result = merger.process_task(item)
        except Exception as exc:
            result = item
            result["recom_result"] = []
            result["stage7_error"] = str(exc)
            unfinished.append({"annotation_id": item.get("annotation_id"), "reason": str(exc)})

        processed.append(result)
        processed_ids.add(_get_task_id(result))
        _checkpoint_stage_output(dataset, "Stage7", output_path, processed, len(stage6_tasks))

    _checkpoint_stage_output(dataset, "Stage7", output_path, processed, len(stage6_tasks), force=True)
    if unfinished:
        save_json(str(output_path.with_name("unfinished_stage7.json")), unfinished)
    return processed


def _build_adjacency(graph_data: List[Dict[str, Any]]) -> Dict[str, set]:
    adjacency = {}
    for link in graph_data:
        adjacency.setdefault(link["source"], set()).add(link["target"])
    return adjacency


def _build_api_repr_dict(tool_desc: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {api["target_action_reprs"]: api for api in tool_desc}


def _calculate_task_metrics(recom_result, actual_apis_reprs):
    predicted = {api["target_action_reprs"] for api in recom_result if api is not None and "target_action_reprs" in api}
    actual = set(actual_apis_reprs)
    precision = len(predicted & actual) / len(recom_result) if recom_result else 0.0
    recall = len(predicted & actual) / len(actual) if actual else 0.0
    exact = 1.0 if precision == 1.0 and recall == 1.0 else 0.0
    return precision, recall, exact


def _calculate_dgd(recom_result, actual_apis_reprs):
    if not actual_apis_reprs:
        return 0.0
    return abs(len(recom_result) - len(actual_apis_reprs)) / len(actual_apis_reprs)


def _calculate_er_core(action_reprs, api_by_repr, adjacency):
    num_pairs = len(action_reprs) - 1
    if num_pairs <= 0:
        return 1.0, False
    executable_pairs = 0
    for idx in range(num_pairs):
        cur = api_by_repr.get(action_reprs[idx])
        nxt = api_by_repr.get(action_reprs[idx + 1])
        if not cur or not nxt:
            continue
        cur_id = cur.get("action_uid")
        nxt_id = nxt.get("action_uid")
        if cur_id in adjacency and nxt_id in adjacency[cur_id]:
            executable_pairs += 1
    return executable_pairs / num_pairs, True


def _calculate_executability_soft(task, api_by_repr, adjacency):
    action_reprs = [step["target_action_reprs"] for step in task.get("recom_result", [])]
    total_steps = len(action_reprs)
    if total_steps <= 1:
        return 1.0
    satisfied = 0
    total = total_steps - 1
    for idx in range(1, total_steps):
        cur = api_by_repr.get(action_reprs[idx])
        if not cur:
            continue
        cur_id = cur.get("action_uid")
        ok = False
        for jdx in range(idx):
            prev = api_by_repr.get(action_reprs[jdx])
            if not prev:
                continue
            if prev.get("action_uid") in adjacency and cur_id in adjacency[prev.get("action_uid")]:
                ok = True
                break
        if ok:
            satisfied += 1
    return satisfied / total if total > 0 else 0.0


def _calculate_soft_ohr(task, api_by_repr, adjacency):
    recom_result = task.get("recom_result", [])
    pred_reprs = [step["target_action_reprs"] for step in recom_result]
    gt_reprs = set(task.get("action_reprs", []))
    if not set(pred_reprs).issubset(gt_reprs):
        return 0.0
    return 1.0 if _calculate_executability_soft(task, api_by_repr, adjacency) == 1.0 else 0.0


def evaluate_predictions(
    predicted_tasks: List[Dict[str, Any]],
    tool_desc_path: str,
    graph_desc_path: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    tool_desc = load_json(tool_desc_path)
    graph_data = load_json(graph_desc_path)
    api_by_repr = _build_api_repr_dict(tool_desc)
    adjacency = _build_adjacency(graph_data)

    total_precision = 0.0
    total_recall = 0.0
    total_exact = 0.0
    total_dgd = 0.0
    total_loose_granularity_acc = 0.0
    total_er = 0.0
    total_ohr = 0.0
    er_count = 0

    iterator = predicted_tasks
    if show_progress:
        iterator = tqdm(predicted_tasks, desc="Evaluating", ncols=100)

    for task in iterator:
        recom_result = task.get("recom_result", [])
        actual_apis_reprs = task.get("action_reprs", [])
        action_reprs = [step["target_action_reprs"] for step in recom_result]

        precision, recall, exact = _calculate_task_metrics(recom_result, actual_apis_reprs)
        total_precision += precision
        total_recall += recall
        total_exact += exact
        total_dgd += _calculate_dgd(recom_result, actual_apis_reprs)
        if abs(len(recom_result) - len(actual_apis_reprs)) <= 1:
            total_loose_granularity_acc += 1.0

        er_score, is_valid = _calculate_er_core(action_reprs, api_by_repr, adjacency)
        if is_valid:
            total_er += er_score
            er_count += 1
        total_ohr += _calculate_soft_ohr(task, api_by_repr, adjacency)

    denom = max(1, len(predicted_tasks))
    return {
        "avg_precision": total_precision / denom,
        "avg_recall": total_recall / denom,
        "avg_dgd": total_dgd / denom,
        "loose_granularity_acc": total_loose_granularity_acc / denom,
        "strict_ohr": total_exact / denom,
        "soft_ohr": total_ohr / denom,
        "real_er": total_er / er_count if er_count > 0 else 0.0,
        "num_tasks": len(predicted_tasks),
        "num_real_er_tasks": er_count,
    }


def _build_dataset_paths(output_root: Path, model_slug: str, dataset: str) -> Dict[str, Path]:
    dataset_dir = output_root / model_slug / dataset
    _ensure_dir(dataset_dir)
    return {
        "dataset_dir": dataset_dir,
        "config": dataset_dir / "config.json",
        "stage1": dataset_dir / "stage1_tasknum.json",
        "stage2": dataset_dir / "stage15_services.json",
        "stage3": dataset_dir / "stage2_dag.json",
        "stage4": dataset_dir / "stage4_chains.json",
        "stage5": dataset_dir / "stage5_candidates.json",
        "stage6": dataset_dir / "stage6_selected.json",
        "stage7": dataset_dir / "stage7_final.json",
        "metrics": dataset_dir / "metrics.json",
        "error": dataset_dir / "dataset_error.json",
    }


def _resolve_split_root(data_root: Path, explicit_root: Optional[str], split_name: str) -> Path:
    if explicit_root:
        return Path(explicit_root)
    split_dir = data_root / split_name
    return split_dir if split_dir.exists() else data_root


def run_dataset(
    dataset: str,
    train_root: Path,
    test_root: Path,
    output_root: Path,
    model_slug: str,
    modules: Dict[str, Any],
    flags: Dict[str, Any],
    resume: bool,
    limit: Optional[int],
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    train_dir = train_root / dataset
    test_dir = test_root / dataset
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing history directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    paths = _build_dataset_paths(output_root, model_slug, dataset)
    save_json(
        str(paths["config"]),
        {
            "dataset": dataset,
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "output_dir": str(paths["dataset_dir"]),
            "limit": limit,
            "flags": flags,
            "llm": {key: value for key, value in llm_config.items() if key != "api_key"},
        },
    )

    stage1_tasks = _run_stage1(dataset, train_dir, test_dir, paths["stage1"], modules, flags, resume, limit)
    stage2_tasks = _run_stage2(dataset, stage1_tasks, train_dir, test_dir, paths["stage2"], modules, flags, resume)
    stage3_tasks = _run_stage3(dataset, stage2_tasks, test_dir, paths["stage3"], modules, flags, resume)
    stage4_tasks = _run_stage4(dataset, stage3_tasks, test_dir, paths["stage4"], modules, resume)
    stage5_tasks = _run_stage5(dataset, stage4_tasks, test_dir, paths["stage5"], modules, flags, resume)
    stage6_tasks = _run_stage6(dataset, stage5_tasks, test_dir, paths["stage6"], modules, flags, resume)
    stage7_tasks = _run_stage7(dataset, stage6_tasks, test_dir, paths["stage7"], modules, resume)

    metrics = evaluate_predictions(
        stage7_tasks,
        str(test_dir / "tool_desc.json"),
        str(test_dir / "graph_desc.json"),
        show_progress=True,
    )
    save_json(str(paths["metrics"]), metrics)
    if paths["error"].exists():
        paths["error"].unlink()
    return {"dataset": dataset, "output_dir": str(paths["dataset_dir"]), "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full DRAG pipeline with a GPT-compatible LLM.")
    parser.add_argument("--data-root", default=DATA_ROOT_DIR)
    parser.add_argument("--history-root", default=HISTORY_ROOT_DIR)
    parser.add_argument("--test-root", default=TEST_ROOT_DIR)
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DIR)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_TO_RUN)
    parser.add_argument("--limit", type=int, default=TASK_LIMIT)
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--gpt-model", default=os.getenv("OPENAI_MODEL") or GPT_MODEL)
    parser.add_argument("--dag-model", default=os.getenv("OPENAI_DAG_MODEL") or GPT_DAG_MODEL)
    parser.add_argument("--path-model", default=os.getenv("OPENAI_PATH_MODEL") or GPT_PATH_MODEL)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or GPT_BASE_URL)
    parser.add_argument("--api-style", default=os.getenv("OPENAI_API_STYLE") or GPT_API_STYLE)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or GPT_API_KEY)
    parser.add_argument(
        "--api-keys",
        default=os.getenv("OPENAI_API_KEYS") or ",".join(GPT_API_KEYS),
        help="Comma-separated API keys for parallel GPT calls.",
    )
    parser.add_argument("--proxy-url", default=os.getenv("OPENAI_PROXY_URL") or GPT_PROXY_URL)
    parser.add_argument("--embedding-url", default=os.getenv("DRAG_EMBEDDING_URL") or EMBEDDING_URL)
    parser.add_argument("--stage3-workers", type=int, default=STAGE3_WORKERS)
    parser.add_argument("--stage6-workers", type=int, default=STAGE6_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["DRAG_EMBEDDING_URL"] = args.embedding_url

    data_root = Path(args.data_root)
    train_root = _resolve_split_root(data_root, args.history_root, "train")
    test_root = _resolve_split_root(data_root, args.test_root, "test")
    output_root = Path(args.output_root)
    _ensure_dir(output_root)

    configured_keys = _split_api_keys(args.api_keys) or _split_api_keys(args.api_key)
    if not configured_keys:
        raise ValueError("No GPT API key configured. Set OPENAI_API_KEY or pass --api-key.")
    _set_active_llm_api_keys(configured_keys)

    flags = deepcopy(FULL_FLAGS)
    flags["stage3_workers"] = min(_normalize_workers(args.stage3_workers), len(_active_llm_api_keys()))
    flags["stage6_workers"] = min(_normalize_workers(args.stage6_workers), len(_active_llm_api_keys()))

    dag_model = args.dag_model or args.gpt_model
    path_model = args.path_model or args.gpt_model
    model_slug = _slugify(args.gpt_model)
    llm_config = {
        "model": args.gpt_model,
        "dag_model": dag_model,
        "path_model": path_model,
        "base_url": args.base_url,
        "api_style": args.api_style,
        "proxy_url": args.proxy_url,
        "stage3_workers": flags["stage3_workers"],
        "stage6_workers": flags["stage6_workers"],
        "api_key_count": len(_active_llm_api_keys()),
    }

    if VERBOSE_PROGRESS:
        print(f"[DRAG] data_root={data_root}")
        print(f"[DRAG] train_root={train_root}")
        print(f"[DRAG] test_root={test_root}")
        print(f"[DRAG] output={output_root / model_slug}")
        print(f"[DRAG] model={args.gpt_model}, dag_model={dag_model}, path_model={path_model}")
        print(f"[DRAG] api_keys={len(_active_llm_api_keys())}, stage3_workers={flags['stage3_workers']}, stage6_workers={flags['stage6_workers']}")

    modules = _load_stage_modules()
    _patch_gpt_runtime(
        modules,
        dag_model=dag_model,
        path_model=path_model,
        base_url=args.base_url,
        api_key=_active_llm_api_keys()[0],
        api_style=args.api_style,
        proxy_url=args.proxy_url,
    )

    results = []
    start_time = time.time()
    for dataset in args.datasets:
        dataset_start = time.time()
        try:
            print(f"\n=== Running DRAG full pipeline: dataset={dataset} ===")
            result = run_dataset(
                dataset=dataset,
                train_root=train_root,
                test_root=test_root,
                output_root=output_root,
                model_slug=model_slug,
                modules=modules,
                flags=flags,
                resume=ENABLE_RESUME and not args.no_resume,
                limit=args.limit,
                llm_config=llm_config,
            )
            result["duration_sec"] = time.time() - dataset_start
            results.append(result)
        except Exception as exc:
            error_payload = {
                "dataset": dataset,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{dataset}] failed: {error_payload['error']}")
            dataset_dir = output_root / model_slug / dataset
            _ensure_dir(dataset_dir)
            save_json(str(dataset_dir / "dataset_error.json"), error_payload)
            results.append(error_payload)
        gc.collect()

    summary_path = output_root / model_slug / "summary.json"
    save_json(
        str(summary_path),
        {
            "llm": llm_config,
            "datasets": args.datasets,
            "duration_sec": time.time() - start_time,
            "results": results,
        },
    )
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
