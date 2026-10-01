# OncoLens

**From-scratch, distributed multi-cancer detection and reporting system (research prototype).**

> ⚠️ **Research prototype. Not a medical device. Not for clinical diagnosis.**
> Stage and grade indicators are model estimates from imaging alone and must be confirmed by a qualified clinician.

OncoLens takes a medical image, routes it to a cancer-specific expert model, and returns a prediction,
severity indicators, a heatmap, calibrated confidence, and a PDF report. Every model is written by hand in
plain PyTorch and trained from random initialization, using parallel GPU strategies (DDP, FSDP, ZeRO-2).

Full specification: [docs/PLAN.md](docs/PLAN.md) · Progress: [docs/PROGRESS.md](docs/PROGRESS.md)

## Modules

| Module | Cancer | Data | Output |
|---|---|---|---|
| A. Skin | Melanoma, BCC, SCC, etc. | ISIC 2019 | 8-class dermoscopy classification + malignancy risk tier |
| B. Brain | Glioma, meningioma, pituitary | Brain Tumor MRI | 4-class MRI classification |
| C. Lung & colon | Adenocarcinoma, SCC | LC25000 (+ NCT-CRC external) | Histopathology tissue classification |
| D. Breast lymph node | Metastasis | PCam-derived Kaggle set | Metastasis present/absent (N-component evidence only) |
| Router | n/a | All of the above | Modality classification + out-of-distribution rejection |

## Research questions

| ID | Question |
|---|---|
| RQ1 | How do hand-built ResNet, ConvNeXt-style, and ViT models compare when trained from scratch on datasets of very different sizes? |
| RQ2 | Does self-supervised pretraining from scratch (SimCLR, MAE) help, especially with 1% and 10% of labels? |
| RQ3 | How do DDP, FSDP, and ZeRO-2 compare in throughput, memory, and scaling efficiency on 2 GPUs? |
| RQ4 | Are from-scratch models well calibrated and robust on external data, and does abstention reduce confident errors? |

## Build roadmap

Each phase ends with a test gate. The next phase starts only when the gate passes.

| Phase | Module | Tests / gate | Status |
|---|---|---|---|
| 0 | Claude Code workspace (CLAUDE.md, settings, progress docs) | Files exist, first commit | ✅ Done |
| 1 | Environment & storage (WSL, GPU, conda, data root on G:) | `scripts/check_env.py` passes | 🔄 In progress (11/12 checks; Kaggle token pending) |
| 2 | Repo scaffold, tooling, CI | ruff, mypy, pytest, CI green | ⬜ |
| 3 | Data acquisition (zipped archives on G:) | Raw manifest + integrity tests | ⬜ |
| 4 | Splits, leakage checks, WebDataset shards, I/O benchmark | Split, leakage, shard, transform tests | ⬜ |
| 5 | Training engine (AMP, DDP, checkpointing, metrics) | Overfit, DDP-equivalence, resume tests | ⬜ |
| 6 | First models from scratch (SimpleCNN, ResNet) | Model unit tests, PCam AUROC above chance | ⬜ |
| 7 | Parallel training on Kaggle 2× T4 + scaling benchmark | DDP vs single parity | ⬜ |
| 8 | ConvNeXt-style across datasets (RQ1) | RQ1 table, 3 seeds | ⬜ |
| 9 | ViT + FSDP / ZeRO-2 (RQ1, RQ3) | FSDP smoke test, scaling v2 | ⬜ |
| 10 | Self-supervised pretraining: SimCLR, MAE (RQ2) | Loss tests, low-label tables | ⬜ |
| 11 | Evaluation & calibration (RQ4) | Metric, calibration, bootstrap tests | ⬜ |
| 12 | Router, Grad-CAM, uncertainty, abstention | Pipeline end-to-end test | ⬜ |
| 13 | Inference optimization (ONNX, INT8, TensorRT) | ONNX parity, INT8 AUROC drop ≤ 1 pt | ⬜ |
| 14 | Report generation (schema, template, validator, PDF) | Validator + PDF tests | ⬜ |
| 15 | FastAPI + Next.js frontend | API tests, e2e smoke | ⬜ |
| 16 | Docker & free deployment | Public URL end to end | ⬜ |
| 17 | Documentation, technical report, release v1.0.0 | Fresh clone reproduces evaluation | ⬜ |

## Storage layout

| Location | Contents |
|---|---|
| `/mnt/g/OncoLens` | This repo (code, configs, docs, small split CSVs) |
| `/mnt/g/oncolens-data` (`$ONCOLENS_ROOT`) | Datasets (zipped), shards, checkpoints, runs, caches, exports |

## License & data

Code: MIT. Each dataset keeps its own license; several are non-commercial (CC-BY-NC), so this project and demo
are non-commercial. Raw datasets are never redistributed.
