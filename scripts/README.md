# scripts/

All scripts read data paths from `configs/default.yaml` or the environment variables `AICUP_DATA_ROOT` (competition
data, read-only) and `AICUP_WORK_ROOT` (scratch tree: exported datasets, runs, prediction caches). Nothing here writes into the
data folders. All scores printed by the evaluation scripts are **proxy** scores, not the official metric.

| script | purpose |
|---|---|
| `audit_dataset.py` | metadata.csv + audit summary, checks the verified counts |
| `make_folds.py` | leak-free 3-fold split (near-duplicate groups) |
| `export_yolo.py`, `make_aug_dataset.py` | labelme -> Ultralytics segmentation dataset (plain / offline-augmented) |
| `train_yolo.py` | config-driven training; `--resume` continues from `last.pt` |
| `watchdog_resume.py` | see below |
| `predict_oof.py` | cache raw predictions (val, shifted val, test) |
| `eval_oof.py`, `compare_models.py`, `diagnose_shift.py` | proxy evaluation and comparisons |
| `make_submission_local.py` | build a submission from cached test predictions, run own validation and the official `--check` |
| `measure_train_cost.py`, `measure_val_cost.py` | GPU memory / speed measurements |
| `visualize.py`, `vis_pred.py` | draw GT or cached predictions |

## watchdog_resume.py

On the 12 GB Windows GPU some training epochs fall into GPU-memory spill (tens of seconds per step while host RAM grows).
A fresh process resuming from `last.pt` is fast again for a few epochs. The watchdog automates that:

* it runs `train_yolo.py --config <cfg> --resume`;
* if no new row is appended to `<work>/runs/<tag>/results.csv` for `--stall-minutes` (default 5), it kills the process
  tree and starts the run again from `last.pt` (at most `--max-attempts`, default 12);
* it stops when `train_yolo.py` exits with code 0 (training finished, `best.pt` copied to `checkpoints/`), or gives up. A crash of `train_yolo.py` (e.g. a CUDA error) is retried from `last.pt` the same way.

`--initial-grace-minutes` (default 25) is the stall limit for the first epoch of every attempt, so a fresh start or a resume is not killed while it
starts up (dataset cache, first epoch incl. validation). `--fold N` overrides the fold of the config. If no `last.pt` exists yet the first attempt starts the run from scratch.
It sets `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:256` (allocator fragmentation settings;
no training hyper-parameter changes). Logs: `<work>/watchdog_logs/`.

```bash
python scripts/watchdog_resume.py --config configs/yolo_eo_aug.yaml --stall-minutes 5
```

Note: a resumed run's `elapsed_s` in `experiments/experiments.csv` only covers the last attempt.
