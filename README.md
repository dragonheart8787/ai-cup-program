# AI CUP 2026 Autumn: solar panel defect detection and root-cause analysis

Status: **skeleton only.** No official files (data, annotations, sample submission,
evaluator, rules) are in this repo yet, so nothing here depends on the official spec.
Task details, scoring formula and submission schema are UNKNOWN / NEED OFFICIAL FILE.
Deadline (as told by the team, not verified): 2026-12-04 23:59:59.

## Layout

| path | purpose |
|---|---|
| `configs/` | YAML configs (`default.yaml`) |
| `data/` | competition data, git-ignored; see `data/README.md` |
| `checkpoints/` | model weights, git-ignored (created on demand) |
| `src/utils/` | config loading + hash, seeding, experiment log, logging |
| `src/postprocess/` | polygon to mask, mask to polygon, polygon validity checks |
| `src/dataset`, `models`, `metrics`, `submission` | empty until the official spec is known |
| `scripts/` | command-line helpers (`check_env.py`) |
| `experiments/` | `experiments.csv` is written here |
| `tests/` | unit tests on synthetic data only |

## Setup (Windows or Linux)

Python 3.11 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
python -m pytest
```

Install torch and the training stack separately, following the notes in `requirements.txt`.
All paths in the code use `pathlib`; there are no shell-only scripts.

## Conventions

- Internal polygon format: `(N, 2)` array of `(x, y)` pixel coordinates; masks are `(H, W)` `uint8` 0/1.
  This is our own convention, unrelated to the official format.
- Every training or evaluation run appends one row to `experiments/experiments.csv`
  via `src.utils.experiments.append_experiment`.
- Licenses of third-party resources are tracked in `LICENSES.md`; read it before adding a dependency.
