# Data Preparation

This repository does not redistribute the full processed datasets. Please download the raw resources from the original sources and preprocess them into the layout expected by the code.

DRAG uses the same processed service-composition datasets as the previous MRG-SC release. The dataset notes below are kept intentionally method-neutral so that the same processed files can be used by DRAG and by the baselines.

## Raw Data Sources

- UltraTool: [JoeYing1019/UltraTool](https://github.com/JoeYing1019/UltraTool)
- JARVIS: [microsoft/JARVIS](https://github.com/microsoft/JARVIS)

## Target Processed Layout

For each dataset variant, prepare:

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

DRAG reads historical examples from `Data/train/<dataset>/` and evaluates on `Data/test/<dataset>/`.

## Required Processed Fields

### `data.json`

Each task record is expected to contain at least:

- `annotation_id`
- `confirmed_task`
- `action_id_list`
- `action_reprs`

Optional metadata fields already supported by the current code include:

- `domain`
- `subdomain`
- `website`

### `tool_desc.json`

Each tool record is expected to contain:

- `action_uid`
- `target_action_reprs`
- `input-type`
- `output-type`

### `graph_desc.json`

Each dependency edge is expected to contain:

- `source`
- `target`
- `type`

Legacy internal versions may include extra statistics such as `count` or `weight`, but they are not required by the public release format.

## Suggested Preprocessing Workflow

1. Download the raw UltraTool and JARVIS resources from the original repositories.
2. Normalize task instances into `data.json`, keeping the natural-language request in `confirmed_task`.
3. Normalize tool metadata into `tool_desc.json`.
4. Normalize dependency edges into `graph_desc.json`.
5. Place each processed dataset under `Data/train/<dataset>/` and `Data/test/<dataset>/`.
6. Run `run_drag_gpt.py` from the repository root.

## Dataset Naming

The current release expects these dataset names:

- `ultratool`
- `mul`
- `hug`
- `daily`

If you derive multiple subsets from JARVIS, map each subset to one of the supported dataset names or extend the dataset list passed to `run_drag_gpt.py`.

## Generated Files

The full DRAG pipeline writes generated stage outputs under the configured output root, for example:

```text
outputs/drag_gpt/<model-name>/<dataset>/
```

Embedding caches such as `*.npy` may be generated inside the local data folders during retrieval. These files are runtime artifacts and are intentionally ignored by git.
