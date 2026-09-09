# DRAG

Anonymous code release.

## Overview

DRAG is a dependency-aware retrieval-augmented graph planning framework for natural-language-driven service composition. Given a user request, DRAG first builds history-augmented planning priors, then asks a GPT-compatible planner to generate a service-aware sub-requirement DAG, decomposes the DAG into ordered chains, and finally composes an executable service workflow through candidate retrieval, hierarchical path selection, and dependency-constrained chain insertion.

The name DRAG stands for **Dependency-aware Retrieval-Augmented Graph planning**.

This repository contains the core code used in the paper, including:

- history-augmented workflow-length prior construction
- historical service-prior retrieval
- GPT-based service-aware DAG planning
- DAG-to-chain decomposition
- constraint-aware service retrieval and filtering
- hierarchical path selection and chain insertion
- evaluation and baseline scripts

The public release does not ship the full processed datasets. Instead, it provides the code, the expected processed data layout, and instructions for preparing data from the original sources.

This repository is released under the MIT License. See [LICENSE](LICENSE).

## Repository Layout

- `DRAG/`: implementation of the DRAG pipeline
  - `Stage1_TaskNumPrediction_train/`: retrieves similar historical tasks and predicts a soft workflow-length prior
  - `Stage2_ServicePreselect_train/`: retrieves historical service priors from similar training tasks
  - `Stage3_TaskDAG/`: generates a service-aware sub-requirement DAG with a GPT-compatible LLM
  - `Stage4_ChainBuild/`: decomposes the DAG into ordered backbone and supplementary chains
  - `Stage5_ServiceRecall_Strict/`: retrieves and filters candidate services for each chain node
  - `Stage6/`: performs beam search and GPT-guided hierarchical path selection
  - `Stage7/`: inserts supplementary chains into the backbone workflow under dependency constraints
  - `Stage8_Evaluate/`: evaluation utilities
  - `gptLLM/`: GPT-compatible LLM caller
  - `localLLM/`: local OpenAI-compatible LLM caller kept for reference
- `Baseline/`: baseline implementations kept for comparison
- `Data/`: expected processed data layout and dataset notes
- `run_drag_gpt.py`: recommended one-command entry point for running the full DRAG pipeline with GPT
- `requirements.txt`: Python dependencies

## Environment Setup

The release has been tested with Python 3.10.

Install dependencies:

```bash
pip install -r requirements.txt
```

DRAG expects:

- a GPT-compatible chat or responses endpoint for Stage3 and Stage6
- an OpenAI-compatible embedding endpoint for retrieval stages
- processed service-composition data following the layout in [Data/README.md](Data/README.md)

For local embeddings, an Ollama-style endpoint such as `http://127.0.0.1:11434/v1/embeddings` can be used.

`faiss-cpu` is optional because the retrieval modules fall back to a NumPy implementation when FAISS is unavailable. `pymilvus` is only needed for the optional DHGL RAG baseline under `Baseline/DHGL/rag`.

## Data Preparation

DRAG uses the same processed datasets and data fields as MRG-SC. Please follow [Data/README.md](Data/README.md) to prepare the datasets from the original UltraTool and JARVIS resources.

The expected layout is:

```text
Data/
  train/
    <dataset>/
      data.json
      tool_desc.json
      graph_desc.json
  test/
    <dataset>/
      data.json
      tool_desc.json
      graph_desc.json
```

Supported dataset names are:

- `daily`
- `hug`
- `mul`
- `ultratool`

The local `Data/train/` and `Data/test/` folders are gitignored by default in this anonymous release.

## Quick Start

The recommended entry point is `run_drag_gpt.py`, which runs the full DRAG pipeline from Stage1 to Stage7 and writes evaluation metrics automatically.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:DRAG_EMBEDDING_URL = "http://127.0.0.1:11434/v1/embeddings"

python .\run_drag_gpt.py `
  --data-root .\Data `
  --output-root .\outputs\drag_gpt `
  --datasets daily hug mul ultratool `
  --gpt-model gpt-5.4-mini `
  --base-url https://api.openai.com/v1 `
  --api-style chat
```

Bash example:

```bash
export OPENAI_API_KEY="your-api-key"
export DRAG_EMBEDDING_URL="http://127.0.0.1:11434/v1/embeddings"

python run_drag_gpt.py \
  --data-root ./Data \
  --output-root ./outputs/drag_gpt \
  --datasets daily hug mul ultratool \
  --gpt-model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 \
  --api-style chat
```

For third-party GPT-compatible gateways, set `--base-url` to the gateway URL. If the gateway uses the OpenAI Responses API, set `--api-style responses`.

## Runtime Configuration

The script can be configured either by editing the configuration block at the top of `run_drag_gpt.py` or by command-line arguments.

Main command-line arguments:

- `--data-root`: root directory containing `train/` and `test/` splits. Default: `./Data`
- `--history-root`: optional explicit history split root. Overrides `<data-root>/train`
- `--test-root`: optional explicit test split root. Overrides `<data-root>/test`
- `--output-root`: output root. Default: `./outputs/drag_gpt`
- `--datasets`: datasets to run, e.g., `daily hug mul ultratool`
- `--limit`: run only the first N test samples for debugging
- `--no-resume`: disable stage-level resume
- `--gpt-model`: shared GPT model name
- `--dag-model`: optional Stage3-only GPT model
- `--path-model`: optional Stage6-only GPT model
- `--base-url`: GPT-compatible API base URL
- `--api-style`: `chat` or `responses`
- `--api-key`: single API key
- `--api-keys`: comma-separated API keys for parallel Stage3 and Stage6 calls
- `--proxy-url`: optional proxy URL
- `--embedding-url`: OpenAI-compatible embedding endpoint
- `--stage3-workers`: worker count for DAG generation
- `--stage6-workers`: worker count for path selection

Relevant environment variables:

- `OPENAI_API_KEY`: single GPT API key
- `OPENAI_API_KEYS`: comma-separated GPT API keys
- `OPENAI_MODEL`: default GPT model
- `OPENAI_DAG_MODEL`: Stage3 GPT model override
- `OPENAI_PATH_MODEL`: Stage6 GPT model override
- `OPENAI_BASE_URL`: GPT API base URL
- `OPENAI_API_STYLE`: `chat` or `responses`
- `OPENAI_PROXY_URL`: optional proxy URL
- `DRAG_EMBEDDING_URL`: embedding endpoint URL
- `DRAG_EMBEDDING_MODEL`: global embedding model override
- `DRAG_STAGE1_EMBEDDING_MODEL`: Stage1 embedding model override
- `DRAG_STAGE2_EMBEDDING_MODEL`: Stage2 embedding model override
- `DRAG_STAGE4_EMBEDDING_MODEL`: Stage4 embedding model override
- `DRAG_STAGE5_EMBEDDING_MODEL`: Stage5 embedding model override
- `DRAG_STAGE6_EMBEDDING_MODEL`: Stage6 embedding model override
- `DRAG_EMBEDDING_TIMEOUT`: embedding timeout in seconds

## Outputs

The output directory follows:

```text
<output-root>/
  <model-name>/
    <dataset>/
      config.json
      stage1_tasknum.json
      stage15_services.json
      stage2_dag.json
      stage4_chains.json
      stage5_candidates.json
      stage6_selected.json
      stage7_final.json
      metrics.json
    summary.json
```

For example, running `gpt-5.4-mini` on `daily` with the default output root produces:

```text
outputs/drag_gpt/gpt-5.4-mini/daily/stage7_final.json
outputs/drag_gpt/gpt-5.4-mini/daily/metrics.json
outputs/drag_gpt/gpt-5.4-mini/summary.json
```

The final workflow prediction is stored in `stage7_final.json` under the `recom_result` field. The aggregated evaluation metrics are stored in `metrics.json` and `summary.json`.

## Baselines

Baseline scripts are kept under `Baseline/` for reference and comparison:

```bash
python Baseline/BIKER.py
python Baseline/Greedy.py
python Baseline/KCAR.py
python Baseline/LLMDirect.py
python Baseline/DHGL/stage1_decompose.py
python Baseline/DHGL/stage2_recall.py
python Baseline/DHGL/stage25_recall_test.py
python Baseline/DHGL/stage3_main.py
```

Some baseline scripts inherit the environment-variable conventions from the previous MRG-SC release. Please check the corresponding script before running a baseline end to end.

## Notes

- API keys should be provided through environment variables or command-line arguments. Do not commit private keys.
- Stage-level outputs are reused by default when `ENABLE_RESUME=True`; pass `--no-resume` to rerun from scratch.
- Embedding caches such as `*.npy` are generated locally and are ignored by git.
- The one-command GPT pipeline in `run_drag_gpt.py` is the recommended reproduction path for DRAG.

## Acknowledgements

This project builds on and benefits from several public research resources. We thank the authors and maintainers of [DHGL](https://github.com/Felicity155/DHGL), [UltraTool](https://github.com/JoeYing1019/UltraTool), and [TaskBench](https://github.com/microsoft/JARVIS/tree/main/taskbench) for making their code, datasets, and benchmarks publicly available to the community.

## Citation

Citation information will be added after the review process.
