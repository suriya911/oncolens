# OncoLens Progress
Current phase: 0 | Mode: GUIDED | Last updated: 2026-09-30

## Phase 0: Workspace
- [x] 0.1 CLAUDE.md  - [x] 0.2 settings.json  - [x] 0.3 PROGRESS.md  - [x] 0.4 research docs  - [x] 0.5 .gitignore/.env.example  - [x] 0.6 commit
- [ ] GATE 0

## Phase 1: Environment & storage
- [ ] 1.1 WSL2 + nvidia-smi  - [ ] 1.2 G: visible at /mnt/g  - [ ] 1.3 .wslconfig (24 GB)  - [ ] 1.4 storage on G: + env vars (write speed: ___ MB/s)
- [ ] 1.5 conda + torch (CUDA, bf16)  - [ ] 1.6 logins (Kaggle, HF, W&B, gh)
- [ ] GATE 1: check_env.py passes

## Phase 2: Repo & GitHub
- [ ] 2.1 scaffold  - [ ] 2.2 storage/seed utils  - [ ] 2.3 pre-commit  - [ ] 2.4 workflows/templates  - [ ] 2.5 first tests  - [ ] 2.6 pushed + branch protection
- [ ] GATE 2: CI green

## Phase 3: Data
- [ ] 3.1 sources verified  - [ ] 3.2 downloads on G:  - [ ] 3.3 manifest
- [ ] GATE 3

## Phase 4: Splits & shards
- [ ] 4.1 splits  - [ ] 4.2 leakage checks  - [ ] 4.3 shards  - [ ] 4.4 SSL pool  - [ ] 4.5 I/O benchmark  - [ ] 4.6 (optional) ext4 VHDX
- [ ] GATE 4

## Phase 5: Trainer
- [ ] 5.1 distributed  - [ ] 5.2 trainer  - [ ] 5.3 strategies  - [ ] 5.4 checkpoint  - [ ] 5.5 metrics  - [ ] 5.6 logging  - [ ] 5.7 train.py  - [ ] 5.8 synthetic data
- [ ] GATE 5: overfit, DDP-equivalence, resume, metrics, accumulation tests green

## Phase 6: First models
- [ ] model tests (M0–M3 shapes/params)  - [ ] 6.1 M0 sanity  - [ ] 6.2 M1 PCam  - [ ] 6.3 M1 Brain MRI
- [ ] GATE 6

## Phase 7: Kaggle DDP
- [ ] 7.1 Kaggle runner  - [ ] 7.2 results fetched to G:  - [ ] 7.3 scaling benchmark v1
- [ ] GATE 7

## Phase 8: ConvNeXt (RQ1)
- [ ] 8.1 ISIC DDP  - [ ] 8.2 other datasets  - [ ] 8.3 imbalance study
- [ ] GATE 8

## Phase 9: ViT + FSDP/ZeRO (RQ1, RQ3)
- [ ] 9.1 ViT runs  - [ ] 9.2 FSDP  - [ ] 9.3 ZeRO-2  - [ ] 9.4 scaling v2
- [ ] GATE 9

## Phase 10: SSL (RQ2, RQ3)
- [ ] 10.1 SimCLR  - [ ] 10.2 batch study  - [ ] 10.3 MAE  - [ ] 10.4 low-label eval
- [ ] GATE 10

## Phase 11: Evaluation (RQ4)
- [ ] test + external evals  - [ ] calibration  - [ ] CIs + tests  - [ ] subgroups  - [ ] failure gallery
- [ ] GATE 11

## Phase 12: Router & XAI
- [ ] 12.1 router  - [ ] 12.2 heatmaps  - [ ] 12.3 TTA + abstention  - [ ] 12.4 pipeline
- [ ] GATE 12

## Phase 13: Inference
- [ ] 13.1 ONNX  - [ ] 13.2 INT8  - [ ] 13.3 TensorRT  - [ ] 13.4 benchmarks
- [ ] GATE 13

## Phase 14: Reports
- [ ] 14.1 schema  - [ ] 14.2 template  - [ ] 14.3 narrator  - [ ] 14.4 validator  - [ ] 14.5 PDF
- [ ] GATE 14

## Phase 15: API & UI
- [ ] 15.1 API  - [ ] 15.2 hardening  - [ ] 15.3 frontend
- [ ] GATE 15

## Phase 16: Deployment
- [ ] 16.1 docker files  - [ ] 16.2 Oracle VM  - [ ] 16.3 VM setup  - [ ] 16.4 secrets  - [ ] 16.5 Vercel  - [ ] 16.6 healthcheck
- [ ] GATE 16

## Phase 17: Docs & release
- [ ] README  - [ ] REPORT.md  - [ ] BENCHMARKS.md  - [ ] model/data cards  - [ ] demo video  - [ ] v1.0.0
- [ ] GATE 17: done

## Open issues
- Repo lives on G: (/mnt/g/OncoLens); git dir on SSD (~/git/oncolens.git); data root /mnt/g/oncolens-data. See LOG.md 2026-09-30.
- GitHub: waiting for user to install `gh` and run `gh auth login`.
