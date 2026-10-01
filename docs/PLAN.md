# OncoLens: Master Build Specification (Research Edition)

**Distributed, from-scratch multi-cancer detection and reporting system**

> **Audience:** this file is written for **Claude Code** (the executor) and for **the user** (the researcher). Claude Code follows it phase by phase, guides the user at every step, and runs the tests that gate each phase.
>
> ⚠️ **Research prototype. Not a medical device. Not for clinical diagnosis.** This disclaimer must appear in the README, the UI, every API response, and every generated report.

---

## Table of Contents

- [Part 0: How to Start (for the user)](#part-0-how-to-start-for-the-user)
- [Part A: Operating Rules for Claude Code](#part-a-operating-rules-for-claude-code)
- [Part B: Project Overview & Research Questions](#part-b-project-overview--research-questions)
- [Part C: Machine, Storage & Compute Facts](#part-c-machine-storage--compute-facts)
- [Part D: Build Phases (step by step, with test gates)](#part-d-build-phases-step-by-step-with-test-gates)
- [Part E: Testing Strategy](#part-e-testing-strategy)
- [Part F: Configuration Reference](#part-f-configuration-reference)
- [Part G: Datasets Reference](#part-g-datasets-reference)
- [Part H: Timeline & Compute Budget](#part-h-timeline--compute-budget)
- [Part I: Risks, Licensing & Ethics](#part-i-risks-licensing--ethics)
- [Appendix 1: CLAUDE.md](#appendix-1-claudemd)
- [Appendix 2: .claude/settings.json](#appendix-2-claudesettingsjson)
- [Appendix 3: docs/PROGRESS.md](#appendix-3-docsprogressmd)
- [Appendix 4: docs/research/EXPERIMENTS.md & LOG.md](#appendix-4-docsresearchexperimentsmd--logmd)

---

## Part 0: How to Start (for the user)

You only do these steps by hand. After that, Claude Code takes over and guides you.

1. **Plug in the My Passport drive (G:)** before opening the Ubuntu (WSL) terminal.
2. Open the Ubuntu terminal and run:
   ```bash
   ls /mnt/g    # must list the drive's contents. If "No such file or directory", see Phase 1, Step 1.2
   mkdir -p ~/projects/oncolens/docs && cd ~/projects/oncolens
   git init
   cp "/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/OncoLens_Master_Plan.md" docs/PLAN.md
   ```
3. Install Claude Code inside WSL if you haven't yet (requires a Pro, Max, Team, Enterprise, or Console account):
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   claude --version
   ```
4. Start Claude Code in the project folder:
   ```bash
   cd ~/projects/oncolens && claude
   ```
5. Give it this first message:
   > Read docs/PLAN.md completely. Then execute Phase 0. After Phase 0, read docs/PROGRESS.md and guide me through the next step.

6. At the start of every later session, say:
   > Read CLAUDE.md and docs/PROGRESS.md, tell me where we are, and continue with the next step.

**Never paste API keys, tokens, or passwords into Claude Code.** You log in to each service yourself with the commands Claude Code gives you.

---

## Part A: Operating Rules for Claude Code

These rules are copied into `CLAUDE.md` in Phase 0 (full text in [Appendix 1](#appendix-1-claudemd)). Summary:

### A.1 Role
You are the user's **research guide and pair programmer**. The user is building this to learn and to showcase skills in model building and parallel GPU training. Explain concepts briefly and precisely before implementing them, and keep the user in control.

### A.2 Step protocol (use for every step)
1. **Goal:** one or two sentences on what this step does and why it matters.
2. **Concept:** a short explanation of the underlying idea (e.g. what a residual connection or all-reduce does), naming the original paper where relevant.
3. **Changes:** list the files you will create or edit, then make the changes.
4. **Commands for the user:** in a code block, labeled `▶ USER RUNS`, with the expected output and what success looks like.
5. **Wait** for the user's output. Diagnose errors before moving on.
6. **Test:** run the step's tests (those you're allowed to run), or hand them over.
7. **Record:** tick the step in `docs/PROGRESS.md`; for experiments, add an entry to `docs/research/LOG.md`. Make a small commit.

### A.3 Execution modes
- **GUIDED (default):** you write code and run only light commands (tests, lint, type checks, git status/commit, quick Python checks). Heavy or sensitive commands go to the user as `▶ USER RUNS`.
- **AUTO:** only if the user explicitly says "switch to AUTO mode". Then you may also run downloads, shard building, and local training yourself, asking before each one. Even in AUTO mode, the user always runs: anything with `sudo`, Windows/PowerShell commands, logins, SSH/SCP, `git push`, Docker, deployment, and anything on Kaggle.

### A.4 Hard rules
- **Phase gates:** never start a phase until the previous phase's gate passes. If a gate fails, debug first.
- **From scratch:** research models use random initialization only. No `pretrained=True`, no torchvision/timm weights, no downloaded checkpoints. The single exception is an experiment explicitly labeled `baseline_imagenet` (optional reference).
- **Hand-written models:** implement architectures in plain PyTorch under `src/oncolens/models/` and `src/oncolens/ssl/`. `timm` is allowed only in `src/oncolens/baselines/`.
- **Storage:** everything large (datasets, shards, checkpoints, caches, runs, ONNX files) goes under `$ONCOLENS_ROOT` on the G: drive. Never write such files into the repo or the home directory. Always get paths from `configs/paths/*.yaml`.
- **Storage check:** every script that touches data or checkpoints must call `oncolens.utils.storage.require_storage()` first, which fails with a clear message if G: isn't mounted or writable.
- **Secrets:** never read, print, or ask for `.env`, `~/.kaggle/`, `~/.ssh/`, `~/.netrc`, or any token. `.env.example` holds variable names only.
- **Library APIs:** before using fast-moving APIs (FSDP, distributed checkpointing, `torch.compile`, pipelining), check the installed version with `python -c "import torch; print(torch.__version__)"` and write code for that version.
- **Honesty:** report results exactly as measured, including negative results. Never tune on the test set.
- **Commits:** small, conventional messages, e.g. `feat(models): add ResNet-18 from scratch`.

---

## Part B: Project Overview & Research Questions

### B.1 What we build
A platform that takes a medical image, routes it to a cancer-specific expert model, and returns a prediction, severity indicators, a heatmap, calibrated confidence, and a PDF report. All models are **designed and trained from scratch**, and training is done with **parallel GPU strategies** (DDP, FSDP, ZeRO-2) on 2 GPUs.

### B.2 Modules & honest scope

| Module | Cancer | Data | Output | Severity / staging-related output |
|---|---|---|---|---|
| A. Skin | Melanoma, BCC, SCC, etc. | ISIC 2019 | 8-class dermoscopy classification | Malignancy risk tier. Melanoma staging needs histopathology; the report says so. |
| B. Brain | Glioma, meningioma, pituitary | Brain Tumor MRI | 4-class MRI classification | Tumor type |
| C. Lung & colon | Adenocarcinoma, SCC | LC25000 (+ NCT-CRC external) | Histopathology tissue classification | Tumor-tissue fraction across tiles |
| D. Breast lymph node | Metastasis | PCam-derived Kaggle set | Metastasis present/absent | Evidence for the **N (node) component** of TNM, never a full stage |
| Router | n/a | All of the above | Modality classification + out-of-distribution rejection | n/a |

True TNM staging requires clinical and pathological data. Every report states: *"Stage and grade indicators are model estimates from imaging alone and must be confirmed by a qualified clinician."*

### B.3 Research questions

| ID | Question | Hypothesis |
|---|---|---|
| **RQ1** | How do hand-built ResNet, ConvNeXt-style, and ViT models compare when trained from scratch on medical datasets of very different sizes (≈7k to ≈220k images)? | CNNs beat ViTs on small datasets; the gap narrows as data grows. |
| **RQ2** | Does self-supervised pretraining from scratch (SimCLR, MAE) on pooled, unlabeled medical training images improve downstream performance, especially with 1% and 10% of labels? | Yes, with the largest gains in low-label settings and on the smallest dataset. |
| **RQ3** | How do DDP, FSDP, and ZeRO-2 compare in throughput, memory, and scaling efficiency on 2 GPUs, and how does global batch size (negatives gathered across GPUs) affect SimCLR? | DDP is fastest for small models; FSDP/ZeRO trade speed for memory; larger SimCLR batches help up to a point. |
| **RQ4** | Are from-scratch models well calibrated and robust on external data, and does abstention reduce high-confidence errors? | Temperature scaling and TTA-based abstention significantly reduce confident errors. |

### B.4 Skills this demonstrates

| Area | Evidence |
|---|---|
| Model building | ResNet, ConvNeXt-style, ViT, SimCLR, MAE written by hand, with unit tests that check shapes, parameter counts, and gradient flow |
| Parallel GPU training | DDP with NCCL, FSDP with activation checkpointing, DeepSpeed ZeRO-2, cross-GPU all-gather for contrastive loss, distributed evaluation |
| Performance engineering | Mixed precision (bf16/fp16), `channels_last`, `torch.compile`, sharded data loading, profiling, scaling benchmarks |
| Research rigor | Patient-level splits, 3 seeds, bootstrap confidence intervals, external validation, calibration, low-label studies, pre-registered experiments |
| Inference & MLOps | ONNX, INT8, TensorRT, FastAPI, Docker, GitHub Actions, free cloud deployment |

### B.5 System architecture

```
Image upload → preprocessing → Router (modality + OOD check)
                                  │
        ┌──────────────┬──────────┴─────┬──────────────────┐
     Skin expert   Brain expert   Histo expert   Lymph-node expert
        └──────────────┴──────────┬─────┴──────────────────┘
                                  ▼
          Temperature scaling + TTA uncertainty + abstention
                                  ▼
                     Grad-CAM / attention heatmap
                                  ▼
                 Structured findings JSON (source of truth)
                                  ▼
       Report: Jinja2 template (+ optional constrained LLM) → HTML/PDF
```

The LLM (optional) only rewrites the findings JSON into plain language. A validator rejects any text that mentions a label, stage, or number not in the JSON and falls back to the template.

---

## Part C: Machine, Storage & Compute Facts

### C.1 Machines

| Machine | Spec | Role |
|---|---|---|
| Laptop | RTX 4070 Laptop GPU (8 GB VRAM, bf16 supported), 32 GB RAM, Windows + WSL2 Ubuntu | Development, tests, single-GPU training (unlimited hours, overnight runs), TensorRT benchmarks |
| Kaggle Notebooks (free) | **2× NVIDIA T4, 16 GB each**, ~30 GPU-hours/week (varies), 12-hour session cap, ~20 GB saved output in `/kaggle/working` | **All parallel (multi-GPU) training and scaling experiments** |

Precision: **bf16** on the RTX 4070; **fp16 with `GradScaler`** on T4 (no efficient bf16 on Turing).

### C.2 Storage layout

| Location | Device | Contents | Approx. size |
|---|---|---|---|
| `~/projects/oncolens` | Internal SSD (WSL ext4) | Git repo (code, configs, docs, small split CSVs) | < 1 GB |
| `~/miniforge3` | Internal SSD | Conda environment | 10–15 GB |
| **`/mnt/g/oncolens`** = **`G:\oncolens`** | **My Passport external drive** | Everything large, listed below | 150–300 GB |

```
/mnt/g/oncolens/                  ← $ONCOLENS_ROOT
├── raw/            downloaded dataset archives (kept zipped)
├── shards/         WebDataset .tar shards (~1 GB each)
├── checkpoints/    model checkpoints, by experiment ID
├── artifacts/      ONNX / INT8 / TensorRT exports, release bundles
├── runs/
│   ├── hydra/      Hydra output folders
│   ├── wandb/      W&B local files
│   └── kaggle/     results downloaded from Kaggle
├── cache/
│   ├── hf/         Hugging Face hub cache (HF_HUB_CACHE)
│   ├── torch/      TORCH_HOME
│   └── pip/        PIP_CACHE_DIR
├── dvc-remote/     DVC storage for data versioning
└── tmp/
```

**Why the code and environment stay on the SSD:** Windows drives are accessed from WSL through a translation layer that is slow for many small files (git, Python imports), and Linux file permissions don't behave normally there. The repo and conda env are small, so they stay on the internal disk.

**Hugging Face token safety:** we set `HF_HUB_CACHE` (cache only), *not* `HF_HOME`, so the login token stays in `~/.cache/huggingface` on the SSD instead of on the external drive.

### C.3 Rules for an external USB hard drive
1. **Never extract datasets into hundreds of thousands of small files on G:.** Keep the downloaded `.zip` archives and stream images directly from them into WebDataset shards. HDDs are fast at large sequential reads and very slow at random small-file reads.
2. **RAM cache:** WSL is given 24 GB RAM (`.wslconfig`). After the first epoch, Linux keeps the shards cached in memory, so later epochs barely touch the drive.
3. **Keep the drive awake and connected:** disable USB selective suspend in Windows power settings, and disable the drive's sleep timer if your drive's utility offers one. A disconnect during training crashes the run; checkpoints every epoch make resuming painless.
4. **Before unplugging:** stop all jobs and run `wsl --shutdown` in PowerShell, then eject the drive in Windows.
5. **Backups:** the external drive is a single point of failure. Best checkpoints are also uploaded to a private Hugging Face model repo.
6. **Optional performance upgrade (only if Phase 4's I/O benchmark shows the GPU starving):** create an ext4 virtual disk on G: and mount it in WSL (Phase 4, Step 4.6).

### C.4 Environment variables (written to `~/.bashrc` in Phase 1)

```bash
export ONCOLENS_ROOT="/mnt/g/oncolens"
export HF_HUB_CACHE="$ONCOLENS_ROOT/cache/hf"
export TORCH_HOME="$ONCOLENS_ROOT/cache/torch"
export PIP_CACHE_DIR="$ONCOLENS_ROOT/cache/pip"
export WANDB_DIR="$ONCOLENS_ROOT/runs/wandb"
```

---

## Part D: Build Phases (step by step, with test gates)

Legend: `▶ USER RUNS` = the user runs it; `⚙ CLAUDE RUNS` = Claude Code may run it in GUIDED mode. Every phase ends with a **Gate**; the next phase starts only when the gate passes.

---

### Phase 0: Bootstrap the Claude Code workspace

**Goal:** create the files that make every future session consistent.

| Step | Action |
|---|---|
| 0.1 | Create `CLAUDE.md` in the repo root from [Appendix 1](#appendix-1-claudemd). |
| 0.2 | Create `.claude/settings.json` from [Appendix 2](#appendix-2-claudesettingsjson). |
| 0.3 | Create `docs/PROGRESS.md` from [Appendix 3](#appendix-3-docsprogressmd). |
| 0.4 | Create `docs/research/EXPERIMENTS.md` and `docs/research/LOG.md` from [Appendix 4](#appendix-4-docsresearchexperimentsmd--logmd). |
| 0.5 | Create `.gitignore` ([Part F.6](#f6-gitignore)) and `.env.example` ([Part F.12](#f12-envexample)). |
| 0.6 | Commit: `chore: bootstrap Claude Code workspace`. |

**Tell the user:** restart Claude Code (`/exit`, then `claude`) so the new settings and CLAUDE.md load.

**Gate 0:** the files exist; `git log` shows the commit; after restart, Claude Code summarizes CLAUDE.md correctly when asked.

---

### Phase 1: Environment & storage

**Step 1.1: Verify WSL and GPU.**
```bash
# ▶ USER RUNS (PowerShell)
wsl -l -v          # Ubuntu must show VERSION 2
wsl --update
```
```bash
# ▶ USER RUNS (Ubuntu)
nvidia-smi         # must show the RTX 4070 Laptop GPU and a driver version
```
If `nvidia-smi` fails: update the NVIDIA driver on Windows (never install a Linux NVIDIA driver inside WSL), then `wsl --shutdown` and reopen.

**Step 1.2: Make sure G: is visible in WSL.**
```bash
# ▶ USER RUNS
ls /mnt/g
```
If missing (drive plugged in after WSL started):
```bash
# ▶ USER RUNS
sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g
```
Also check the drive's file system in PowerShell: `Get-Volume -DriveLetter G`. NTFS is ideal; exFAT works but has no symlinks. **Do not reformat the drive**: it may contain your files.

**Step 1.3: WSL memory settings.** Claude Code writes `setup/wslconfig.txt` ([Part F.10](#f10-wslconfig)).
```powershell
# ▶ USER RUNS (PowerShell)
Copy-Item "\\wsl$\Ubuntu\home\<LINUX_USER>\projects\oncolens\setup\wslconfig.txt" "$env:USERPROFILE\.wslconfig"
wsl --shutdown
```
(Adjust the distro name if `wsl -l -v` shows something other than `Ubuntu`.) Reopen Ubuntu and verify with `free -g` (total ≈ 24).

**Step 1.4: Create project storage on G:.** Claude Code writes `setup/storage_setup.sh` ([Part F.11](#f11-setupstorage_setupsh)).
```bash
# ▶ USER RUNS
bash setup/storage_setup.sh /mnt/g/oncolens
source ~/.bashrc
echo $ONCOLENS_ROOT && ls $ONCOLENS_ROOT
```
Expected: the folder tree from Part C.2 and a write-speed line (USB HDDs typically show roughly 80–150 MB/s; record the number in `docs/research/LOG.md`).

**Step 1.5: Conda environment.**
```bash
# ▶ USER RUNS
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b && ~/miniforge3/bin/conda init && exec bash
cd ~/projects/oncolens
conda env create -f environment.yml
conda activate oncolens
```
Then install PyTorch with the **current command from pytorch.org** (Linux, pip, latest CUDA 12.x build), for example:
```bash
# ▶ USER RUNS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12X   # replace with pytorch.org's current command
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```
Expected: version string, `True`, `NVIDIA GeForce RTX 4070 Laptop GPU`, `True`.

**Step 1.6: Accounts & logins** (user creates free accounts: GitHub, Kaggle, Hugging Face, Weights & Biases; later Vercel and Oracle Cloud).
```bash
# ▶ USER RUNS
# Kaggle: kaggle.com → Settings → API → Create New Token (downloads kaggle.json to Windows Downloads)
mkdir -p ~/.kaggle && cp "/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/Downloads/kaggle.json" ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
huggingface-cli login        # paste a token with write access when prompted
wandb login
sudo apt install -y gh && gh auth login
```

**Tests (⚙ CLAUDE RUNS):** `python scripts/check_env.py`, which prints and asserts: CUDA available, bf16 supported, `ONCOLENS_ROOT` set, writable, on `/mnt/g`, free space ≥ 150 GB, Kaggle/HF/W&B credential files exist (checks existence only, never reads contents).

**Gate 1:** `check_env.py` passes all checks.

---

### Phase 2: Repository scaffold, tooling & GitHub

| Step | Action |
|---|---|
| 2.1 | Create the repo structure ([Part F.1](#f1-repository-structure)), `pyproject.toml`, `environment.yml`, `configs/` (Hydra), `src/oncolens/__init__.py`. |
| 2.2 | Create `src/oncolens/utils/storage.py` with `require_storage()` and `paths` helpers; `src/oncolens/utils/seed.py` (seed everything, per-rank seeds). |
| 2.3 | Add `.pre-commit-config.yaml` ([F.7](#f7-pre-commit-configyaml)); user runs `pre-commit install`. |
| 2.4 | Add GitHub workflows ([F.8](#f8-github-workflows)), PR/issue templates, Dependabot, CODEOWNERS. |
| 2.5 | Write first tests: `tests/test_storage.py`, `tests/test_config.py`. |
| 2.6 | User creates the GitHub repo and pushes. |

```bash
# ▶ USER RUNS
pip install -e ".[dev]"
pre-commit install
gh repo create oncolens --public --source=. --remote=origin
git push -u origin main
```

**GitHub settings (user, in the browser):** protect `main` (require PR + passing `ci` check, block force-push); enable Dependabot alerts, secret scanning, CodeQL; add topics (`medical-imaging`, `distributed-training`, `pytorch`, `ddp`, `fsdp`, `self-supervised-learning`).

**Tests (⚙ CLAUDE RUNS):** `ruff check .`, `mypy src/oncolens`, `pytest`.
`test_config.py` asserts Hydra composes for `paths=local`, `paths=kaggle`, `paths=ci`, and that no configured data path points inside the repo.

**Gate 2:** all local checks pass, and the `ci` workflow is green on GitHub.

---

### Phase 3: Data acquisition

**Step 3.1: Verify sources.** Claude Code presents the dataset table ([Part G](#part-g-datasets-reference)) with links, licenses, and access requirements, and asks the user to accept competition rules on Kaggle where needed (the "Histopathologic Cancer Detection" competition requires joining).

**Step 3.2: Download to G: (kept zipped).** Claude Code writes `scripts/download_data.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
: "${ONCOLENS_ROOT:?ONCOLENS_ROOT not set}"
RAW="$ONCOLENS_ROOT/raw"; mkdir -p "$RAW"/{isic2019,brain_mri,lc25000,pcam,nct_crc}
# Verify slugs on kaggle.com first. No --unzip: archives stay zipped (HDD-friendly).
kaggle datasets download -d andrewmvd/isic-2019 -p "$RAW/isic2019"
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p "$RAW/brain_mri"
kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images -p "$RAW/lc25000"
kaggle competitions download -c histopathologic-cancer-detection -p "$RAW/pcam"
echo "NCT-CRC-HE-100K and CRC-VAL-HE-7K: download the zips from Zenodo into $RAW/nct_crc (link in PLAN Part G)"
```
```bash
# ▶ USER RUNS (takes a while; keep the laptop awake and the drive connected)
bash scripts/download_data.sh
```

**Step 3.3: Integrity manifest.** `scripts/verify_raw.py` lists each archive, its size, SHA-256, file count, and class counts (read from inside the zip, without extracting) into `data/manifests/raw_manifest.json` (committed to git).

**Tests (⚙ CLAUDE RUNS):** `pytest tests/test_raw_manifest.py`: archives open, counts match expected ranges (e.g. ISIC 2019 ≈ 25k training images, PCam set ≈ 220k patches, Brain MRI ≈ 7k), no zero-byte files.

**Gate 3:** the manifest exists and all integrity tests pass.

---

### Phase 4: Splits, leakage checks, shards & I/O benchmark

**Step 4.1: Patient-level splits.** `scripts/make_splits.py` writes `data/splits/<dataset>/{train,val,test}.csv` (70/15/15, stratified, grouped by patient/lesion ID when available). External sets go to `data/splits/<dataset>/external.csv`. Splits are fixed with seed 42 and versioned in git (small CSVs).

**Step 4.2: Leakage checks.** `scripts/check_leakage.py`:
- no shared patient/lesion IDs across splits;
- **perceptual-hash near-duplicate detection** across splits (LC25000 contains augmented near-duplicates: duplicates must stay in the same split);
- HAM10000 ↔ ISIC 2019 overlap removed from the external set.

**Step 4.3: Shards.** `scripts/build_shards.py` streams images **straight from the zip archives** into WebDataset shards on G: (`$ONCOLENS_ROOT/shards/<dataset>/<split>-%05d.tar`, ~1 GB each), storing image bytes (re-encoded at a fixed max resolution per dataset: 512 px for ISIC, 256 px for MRI, native size for histology patches) plus a JSON label record.
```bash
# ▶ USER RUNS
python scripts/build_shards.py dataset=pcam
python scripts/build_shards.py dataset=isic2019
python scripts/build_shards.py dataset=brain_mri
python scripts/build_shards.py dataset=lc25000
python scripts/build_shards.py dataset=nct_crc
```

**Step 4.4: Unlabeled SSL pool.** A shard index `ssl_pool` that references **only training-split** images from all datasets (≈ 270k images). Validation and test images are never used for self-supervised pretraining.

**Step 4.5: I/O benchmark.**
```bash
# ▶ USER RUNS
python scripts/io_benchmark.py dataset=isic2019 num_workers=2,4,8 epochs=2
```
It reports images/sec for epoch 1 (cold, from disk) and epoch 2 (warm, RAM cache), and compares against the model's GPU throughput measured on synthetic data. Record results in the LOG.

**Step 4.6 (only if the GPU would starve):** optional ext4 virtual disk on G:
```powershell
# ▶ USER RUNS (PowerShell as administrator)
diskpart
#   create vdisk file="G:\oncolens-vhd\oncolens.vhdx" maximum=307200 type=expandable
#   exit
wsl --mount --vhd G:\oncolens-vhd\oncolens.vhdx --bare
```
```bash
# ▶ USER RUNS (Ubuntu) — identify the NEW disk carefully with lsblk; formatting the wrong disk destroys data
lsblk
sudo mkfs.ext4 /dev/sdX
```
```powershell
# ▶ USER RUNS (PowerShell as administrator)
wsl --unmount G:\oncolens-vhd\oncolens.vhdx
wsl --mount --vhd G:\oncolens-vhd\oncolens.vhdx --name oncolens
```
Then `sudo chown -R $USER /mnt/wsl/oncolens`, move the shards there, and set `ONCOLENS_ROOT=/mnt/wsl/oncolens` in `~/.bashrc`. This mount must be repeated after each reboot or reconnect. Skip this step unless the benchmark demands it.

**Tests (⚙ CLAUDE RUNS):**
- `test_splits.py`: split ratios within ±1%, class stratification within ±2%, zero ID overlap.
- `test_leakage.py`: zero cross-split near-duplicates above the hash threshold.
- `test_shards.py`: round trip (image decoded from a shard equals the source image after the same resize), label counts in shards equal split CSV counts, every rank gets disjoint shards when `world_size=2`.
- `test_transforms.py`: output shape/dtype/range, deterministic under a fixed seed, stain augmentation keeps values in range.

**Gate 4:** all tests pass; warm-epoch loading throughput ≥ GPU throughput for the planned model (or Step 4.6 applied).

---

### Phase 5: Training engine (from scratch, no Lightning)

**Goal:** a hand-written, distributed-ready trainer that every later model uses.

| Step | File | Content |
|---|---|---|
| 5.1 | `training/distributed.py` | `setup_distributed()` (NCCL on GPU, gloo on CPU), `cleanup()`, `is_main()`, `all_gather_tensors()`, `reduce_mean()` |
| 5.2 | `training/trainer.py` | Loop with AMP (bf16/fp16 auto), `GradScaler`, gradient accumulation with `no_sync()`, gradient clipping, EMA, cosine schedule with warmup, LR scaling `lr = base_lr × global_batch / 256` |
| 5.3 | `training/strategies.py` | `wrap_model(model, strategy)` for `single`, `ddp`, `fsdp` (DeepSpeed in its own entry script) |
| 5.4 | `training/checkpoint.py` | Save/resume model, optimizer, scheduler, scaler, EMA, epoch, step, RNG states (per rank); atomic writes (temp file + rename) so an unplugged drive can't corrupt the last good checkpoint |
| 5.5 | `training/metrics.py` | Distributed-correct metrics: gather predictions from all ranks, drop `DistributedSampler` padding duplicates, then compute AUROC etc. |
| 5.6 | `training/logging.py` | Rank-0 W&B + console logging; logs world size, global batch, precision, git commit, config |
| 5.7 | `scripts/train.py` | Hydra entry point; calls `require_storage()` first |
| 5.8 | `data/synthetic.py` | Synthetic dataset for tests and CI |

Core distributed setup (reference):
```python
import os, torch, torch.distributed as dist

def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank, local_rank, world = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world
```

Training-loop essentials (reference):
```python
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and cfg.trainer.prefer_bf16 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
for epoch in range(start_epoch, cfg.trainer.epochs):
    sampler.set_epoch(epoch)
    for step, (x, y) in enumerate(loader):
        sync = (step + 1) % cfg.trainer.grad_accum == 0
        ctx = model.no_sync() if (not sync and hasattr(model, "no_sync")) else nullcontext()
        with ctx, torch.autocast("cuda", dtype=amp_dtype):
            loss = criterion(model(x), y) / cfg.trainer.grad_accum
        scaler.scale(loss).backward()
        if sync:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step(); ema.update(model)
```

**Tests (⚙ CLAUDE RUNS):**
- `test_overfit_one_batch.py`: a small model reaches near-zero loss on one fixed batch within N steps (catches broken loss/labels/optimizer).
- `test_ddp_equivalence.py`: with 2 CPU ranks (gloo), gradients after one step equal the single-process gradients on the concatenated batch (tolerance 1e-5).
- `test_checkpoint_resume.py`: train 3 steps, save, resume, train 1 step → identical loss to an uninterrupted 4-step run.
- `test_distributed_metrics.py`: AUROC from 2 gathered ranks equals AUROC on the full set (sampler padding removed).
- `test_grad_accum.py`: accumulation over 4 micro-batches equals one big batch (within tolerance).
- Smoke run: `torchrun --standalone --nproc_per_node=2` on CPU with synthetic data for 20 steps (run via pytest wrapper `tests/test_smoke_ddp.py`).

**Gate 5:** all trainer tests pass locally and in CI.

---

### Phase 6: First models from scratch (single GPU)

#### 6.A Model specifications (all hand-written, random init)

| ID | Model | Key components to implement | Target params (for unit test) |
|---|---|---|---|
| M0 | SimpleCNN | conv-BN-ReLU blocks, global pooling | ~1M |
| M1 | ResNet-18 / ResNet-34 | BasicBlock, residual shortcut with 1×1 projection, stem, zero-init last BN γ | ResNet-18 ≈ 11.7M at 1000 classes; ResNet-34 ≈ 21.8M |
| M2 | ConvNeXt-style (Nano/Tiny) | 7×7 depthwise conv, LayerNorm (channels-last), inverted bottleneck MLP, GELU, layer scale, stochastic depth | ConvNeXt-T ≈ 28.6M at 1000 classes; a smaller "Nano" config for small datasets |
| M3 | ViT-Tiny / ViT-Small | patch embedding (conv), class token, learnable position embedding, multi-head attention via `F.scaled_dot_product_attention`, pre-norm blocks, MLP, stochastic depth | ViT-S/16 ≈ 22M; ViT-Ti/16 ≈ 5.7M |
| M4 | SimCLR | encoder + 2-layer projection head, NT-Xent loss with cross-GPU negatives | encoder-dependent |
| M5 | MAE | random patch masking (75%), ViT encoder on visible patches, light decoder, pixel-reconstruction loss on masked patches | encoder-dependent |
| R | Router | small CNN (M0-sized) | ~1M |

Initialization: Kaiming-normal for convolutions; truncated-normal (std 0.02) for ViT linear layers and embeddings; zeros for biases.

From-scratch training recipe (defaults, tuned per dataset on validation only): AdamW, weight decay 0.05, warmup 5–10 epochs, cosine decay, label smoothing 0.1, EMA, RandAugment; Mixup/CutMix for ViTs; 100–300 epochs on small datasets, 20–40 on PCam.

**Model unit tests (⚙ CLAUDE RUNS)** in `tests/test_models.py`: output shape for several input sizes, parameter count within ±1% of the reference figure, every parameter receives a gradient, no NaNs under fp16/bf16 autocast, deterministic forward with a fixed seed, `torch.compile` compatibility (marked slow).

#### 6.B Steps

| Step | Experiment | Data | Where |
|---|---|---|---|
| 6.1 | M0 sanity: overfit 1 batch, then 3-epoch run; LR range test | PCam | Laptop |
| 6.2 | **M1 ResNet-18 from scratch**, full training | PCam | Laptop (overnight) |
| 6.3 | M1 on Brain MRI (smallest dataset; expect overfitting → study augmentation strength) | Brain MRI | Laptop |

```bash
# ▶ USER RUNS (example)
python scripts/train.py experiment=e001_m0_pcam_sanity
python scripts/train.py experiment=e002_m1_resnet18_pcam seed=0
```

**Gate 6:** M0 overfits one batch; M1 PCam validation AUROC clearly above chance and still improving at the end of warmup; results logged in LOG.md with the W&B link.

---

### Phase 7: First parallel training on Kaggle (2× T4) + scaling benchmark

**Step 7.1: Kaggle runner.** Claude Code writes `notebooks/kaggle/run_experiment.ipynb`, with cells that:
1. read `HF_TOKEN` and `WANDB_API_KEY` from **Kaggle Secrets** (user adds them in the notebook: Add-ons → Secrets);
2. clone the GitHub repo and `pip install -e .`;
3. run `torchrun --standalone --nproc_per_node=2 scripts/train.py experiment=<ID> paths=kaggle`;
4. upload checkpoints and metrics to the private Hugging Face repo `<HF_USER>/oncolens-checkpoints`.

**User steps on Kaggle:** create a notebook → upload/import the `.ipynb` → Settings → Accelerator → **GPU T4 ×2** → attach the competition/dataset inputs → add secrets → **Save Version → Save & Run All (Commit)** so it runs in the background.

**Step 7.2: Download results to G:.**
```bash
# ▶ USER RUNS
python scripts/fetch_results.py experiment=e010_m1_resnet18_pcam_ddp   # pulls from HF Hub into $ONCOLENS_ROOT/runs/kaggle/
```

**Step 7.3: Scaling benchmark v1** (`scripts/benchmark_scaling.py`, 50 warmup + 300 timed steps per config):

| Run | GPUs | Strategy | Precision | Global batch | Metrics |
|---|---|---|---|---|---|
| S1 | 1× T4 | single | fp16 | 256 | images/s, step time, peak VRAM, GPU util |
| S2 | 2× T4 | DDP | fp16 | 512 | same + scaling efficiency = S2 / (2 × S1) |
| S3 | 2× T4 | DDP + `channels_last` | fp16 | 512 | same |
| S4 | 2× T4 | DDP + `torch.compile` | fp16 | 512 | same + compile time |
| S5 | 2× T4 | DDP, `bucket_cap_mb` 25 vs 100 | fp16 | 512 | communication overhead (PyTorch profiler trace) |
| S6 | RTX 4070 | single | bf16 | 256 (accum.) | same |

**Tests:**
- Parity: final validation AUROC of DDP (2 GPUs) vs single GPU at equal global batch is within seed noise (compare 2 seeds each).
- `test_kaggle_paths.py` (⚙): `paths=kaggle` resolves to `/kaggle/input` and `/kaggle/working`.

**Gate 7:** a DDP run completed on 2× T4, results fetched to G:, scaling table and chart committed to `docs/BENCHMARKS.md`.

---

### Phase 8: ConvNeXt-style from scratch across datasets (RQ1)

| Step | Experiment |
|---|---|
| 8.1 | M2 on ISIC 2019, DDP on Kaggle (main skin model) |
| 8.2 | M2 and M1 on Brain MRI, LC25000, and NCT-CRC (laptop or Kaggle depending on size) |
| 8.3 | Class imbalance study on ISIC: weighted sampler vs focal loss vs plain CE (balanced accuracy, melanoma sensitivity) |

**Gate 8:** RQ1 table filled for M1 and M2 on all datasets with 3 seeds each (mean ± std).

---

### Phase 9: ViT from scratch + sharded training (RQ1, RQ3)

| Step | Experiment |
|---|---|
| 9.1 | M3 ViT-Ti and ViT-S from scratch on PCam and ISIC with strong augmentation; compare to M1/M2 |
| 9.2 | **FSDP** on 2× T4: ViT-S and a larger ViT-B config; `FULL_SHARD` vs `SHARD_GRAD_OP`; activation checkpointing on/off |
| 9.3 | **DeepSpeed ZeRO-2** on 2× T4 for the same model |
| 9.4 | Scaling benchmark v2: DDP vs FSDP vs ZeRO-2 for ViT-S and ViT-B (throughput, peak memory per GPU, max batch that fits) |

FSDP config (reference, `configs/trainer/fsdp.yaml`):
```yaml
strategy: fsdp
sharding: FULL_SHARD
activation_checkpointing: true
cpu_offload: false
prefer_bf16: true       # falls back to fp16 on T4
batch_size_per_gpu: 64
grad_accum: 1
```

DeepSpeed config (reference, `configs/trainer/deepspeed_z2.json`):
```json
{
  "train_micro_batch_size_per_gpu": 64,
  "gradient_accumulation_steps": 1,
  "fp16": { "enabled": true },
  "zero_optimization": { "stage": 2, "overlap_comm": true, "contiguous_gradients": true },
  "gradient_clipping": 1.0
}
```

**Tests:**
- ⚙ `test_fsdp_smoke.py`: 2 CPU ranks (gloo), FSDP wraps the ViT, one step runs, a full-state-dict checkpoint saves and reloads into a plain model with identical outputs. (Mark as skipped automatically if the installed PyTorch version doesn't support FSDP on CPU/gloo, and run it on Kaggle instead.)
- Kaggle: FSDP and DDP losses track each other over the first 200 steps for the same seed and global batch.

**Gate 9:** RQ1 complete for M1–M3; RQ3 table and charts (DDP vs FSDP vs ZeRO-2) in `BENCHMARKS.md`.

---

### Phase 10: Self-supervised pretraining from scratch (RQ2, RQ3)

**Step 10.1: Distributed SimCLR** (ResNet encoder) on the `ssl_pool`. Negatives are gathered from **all GPUs**, which is why multi-GPU training matters for contrastive learning:

```python
import torch, torch.nn.functional as F, torch.distributed as dist
from torch.distributed.nn.functional import all_gather   # all_gather that passes gradients back

def nt_xent(z1, z2, temperature=0.2):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    if dist.is_available() and dist.is_initialized():
        z1_all, z2_all, rank = torch.cat(all_gather(z1)), torch.cat(all_gather(z2)), dist.get_rank()
    else:
        z1_all, z2_all, rank = z1, z2, 0
    b = z1.size(0)
    labels = torch.arange(b, device=z1.device) + rank * b
    self_mask = F.one_hot(labels, z1_all.size(0)).bool()
    def side(a, a_all, other_all):
        pos_and_neg = a @ other_all.T / temperature
        same_view = (a @ a_all.T / temperature).masked_fill(self_mask, float("-inf"))
        return F.cross_entropy(torch.cat([pos_and_neg, same_view], dim=1), labels)
    return 0.5 * (side(z1, z1_all, z2_all) + side(z2, z2_all, z1_all))
```

**Step 10.2: Batch-size study:** global batch 256 / 512 / 1024 (via per-GPU batch × 2 GPUs, with accumulation where needed; note that gradient accumulation does *not* add negatives, which is itself a finding to report).

**Step 10.3: MAE** (ViT-S encoder) on the `ssl_pool` with FSDP.

**Step 10.4: Downstream evaluation:** linear probe and full fine-tuning on each dataset with **1%, 10%, 100%** of training labels (label subsets fixed by seed and saved in `data/splits/`), compared against from-scratch supervised training at equal fine-tuning budget. Optional: one `baseline_imagenet` reference row.

**Tests (⚙ CLAUDE RUNS):**
- `test_simclr_loss.py`: loss equals a hand-computed value on tiny inputs; 2-rank (gloo) loss equals single-process loss on the concatenated batch; gradients flow to both views.
- `test_mae.py`: masking ratio exact, visible/masked indices disjoint and complete, reconstruction loss computed only on masked patches, unshuffle restores order.
- `test_label_subsets.py`: subsets are nested (1% ⊂ 10% ⊂ 100%) and stratified.

**Gate 10:** RQ2 tables (linear probe and fine-tune, 3 label fractions, 3 seeds) and the SimCLR batch-size chart are committed.

---

### Phase 11: Evaluation & research analysis (RQ4)

For each final model, on the test set **once** and on external sets:
- macro AUROC, per-class AUROC, balanced accuracy, per-class sensitivity/specificity, F1;
- 95% bootstrap confidence intervals (1,000 resamples); paired bootstrap tests for model comparisons;
- calibration: reliability diagrams, ECE before/after temperature scaling (temperature fit on validation);
- operating points chosen on validation (e.g. ≥ 95% sensitivity for malignant classes);
- subgroup metrics where metadata exists (ISIC: age group, sex, anatomic site);
- failure gallery: top false negatives with heatmaps.

```bash
# ▶ USER RUNS
python scripts/evaluate.py experiment=<ID> split=test
python scripts/evaluate.py experiment=<ID> split=external
```

**Tests (⚙):** `test_metrics.py` (metrics match scikit-learn on fixed inputs), `test_calibration.py` (ECE of a perfectly calibrated synthetic model ≈ 0; temperature scaling lowers NLL on an overconfident synthetic model), `test_bootstrap.py` (CI contains the true value on synthetic data at the expected rate, approximately).

**Gate 11:** `reports/` contains complete evaluation folders for every final model; results tables in `docs/research/REPORT.md` are filled.

---

### Phase 12: Router, explainability, uncertainty, abstention

| Step | Action |
|---|---|
| 12.1 | Train the router from scratch on all modalities + an "other" class built from random non-medical images with permissive licenses (for OOD rejection) |
| 12.2 | Grad-CAM for CNNs and attention rollout for ViTs (implemented by hand with hooks) |
| 12.3 | TTA uncertainty (8 flips/rotations) and an abstention rule tuned on validation |
| 12.4 | `src/oncolens/pipeline.py`: one function from image bytes to `Findings` |

**Tests (⚙):** router accuracy on a held-out mixed set; OOD images are rejected above a set rate; heatmaps have the input's spatial size and are non-constant; abstention lowers the error rate among accepted predictions on validation (selective-risk curve); pipeline end-to-end test on one sample per module.

**Gate 12:** the pipeline returns valid `Findings` for every module, and the selective-risk curve is in the report.

---

### Phase 13: Inference optimization

| Step | Action |
|---|---|
| 13.1 | ONNX export with a dynamic batch axis; `onnx.checker` |
| 13.2 | INT8 static quantization (ONNX Runtime) with a calibration subset |
| 13.3 | TensorRT FP16 engine on the RTX 4070 (`trtexec`) |
| 13.4 | `scripts/benchmark_inference.py`: p50/p95 latency at batch 1 and throughput at batch 16 for PyTorch eager, PyTorch compiled, ONNX Runtime CUDA, TensorRT, ONNX Runtime CPU INT8 |

**Tests (⚙):** `test_onnx_parity.py` (max abs difference vs PyTorch < 1e-3 in fp32), INT8 AUROC drop ≤ 1 point on validation (else exclude that model from INT8), output shapes for batch 1 and 16.

**Gate 13:** inference table in `BENCHMARKS.md`; release bundle in `$ONCOLENS_ROOT/artifacts/release/` and uploaded to the public model repo `<HF_USER>/oncolens-models`.

---

### Phase 14: Report generation

| Step | Action |
|---|---|
| 14.1 | `reporting/schema.py`: Pydantic `Findings` model (modality, module, model version, probabilities, risk tier, severity indicators, confidence, uncertainty, abstained, heatmap, limitations, disclaimer) |
| 14.2 | Jinja2 HTML template: Summary, Image + heatmap, Probability table, Risk & severity indicators, Confidence & uncertainty, Limitations, Suggested next steps (generic), Model & data versions, Disclaimer |
| 14.3 | Optional narrator: a small quantized instruction model served by `llama.cpp` receives only the JSON |
| 14.4 | Validator: rejects narrator output mentioning labels, stages, or numbers not in the JSON; falls back to template text |
| 14.5 | PDF rendering with WeasyPrint |

**Tests (⚙):** schema validation (rejects probabilities outside [0,1], missing disclaimer); `test_report_validator.py` (rejects "stage III" when no stage is in the JSON, rejects invented percentages, accepts faithful paraphrase); template renders for every module and for the abstained case; PDF file is produced and contains the disclaimer text.

**Gate 14:** sample PDFs for every module saved in `docs/samples/` (using only redistributable images).

---

### Phase 15: API & frontend

| Step | Action |
|---|---|
| 15.1 | FastAPI: `GET /health`, `POST /v1/analyze`, `GET /v1/report/{id}`, `GET /v1/report/{id}.pdf`, `GET /v1/models` |
| 15.2 | Upload limits (10 MB), MIME/type validation, rate limiting, CORS restricted to the frontend origin, images processed in memory and not stored |
| 15.3 | Frontend (Next.js): upload, sample gallery, result card, heatmap overlay slider, probability bars, PDF download, permanent disclaimer banner |

**Tests (⚙):** `test_api.py` with FastAPI `TestClient` (health, valid upload, oversize upload → 413, wrong type → 415, abstain path, PDF content type); frontend lint + build; Playwright smoke test (upload sample → result visible) (marked e2e).

**Gate 15:** `docker compose up` locally serves the full app, and the e2e smoke test passes.

---

### Phase 16: Docker & free deployment

Topology: **frontend on Vercel (Hobby)** → **backend on an Oracle Cloud Always Free Arm VM** (FastAPI + ONNX Runtime INT8 on CPU behind Caddy with automatic HTTPS) → models pulled from the Hugging Face Hub. Docker images built by GitHub Actions for amd64 + arm64 and stored in GitHub Container Registry.

Free-tier facts as checked in September 2026 (these change; re-check before deploying):
- Oracle's Always Free Arm allowance for new free-tier accounts should be planned as **2 OCPU / 12 GB RAM** (reduced from 4 OCPU / 24 GB in mid-2026). Signup requires card verification; regional capacity can be limited.
- Hugging Face **Spaces running Gradio/Docker now require a paid plan** for new Spaces (static Spaces remain free), so Spaces is not used for the backend.
- Vercel Hobby is free for non-commercial use.

**Plan B (zero servers):** run the INT8 ONNX models in the browser with ONNX Runtime Web and host only static files on Vercel or GitHub Pages ("privacy mode": images never leave the device; template report, no LLM).

| Step | Who | Action |
|---|---|---|
| 16.1 | Claude | `deploy/Dockerfile`, `deploy/requirements-serve.txt`, `deploy/docker-compose.yml`, `deploy/Caddyfile` ([F.9](#f9-deployment-files)) |
| 16.2 | User | Oracle console: create Always Free account; VM `VM.Standard.A1.Flex`, Ubuntu 24.04 aarch64, within free limits; upload SSH public key; open TCP 80/443 in the VCN security list |
| 16.3 | User | Run `deploy/oracle_setup.sh` on the VM; copy compose/Caddyfile; create `.env` on the VM (`HF_TOKEN`, `API_DOMAIN=<VM-IP>.sslip.io`) |
| 16.4 | User | `gh secret set ORACLE_HOST`, `ORACLE_USER`, `ORACLE_SSH_KEY`, `HF_TOKEN` |
| 16.5 | User | Vercel: import the GitHub repo with root directory `frontend`, set `NEXT_PUBLIC_API_URL` |
| 16.6 | Claude | Scheduled GitHub Action that calls `/health` daily (keeps the VM active and alerts on failure) |

```bash
# ▶ USER RUNS (on your laptop)
ssh-keygen -t ed25519 -f ~/.ssh/oracle_oncolens
scp -i ~/.ssh/oracle_oncolens deploy/docker-compose.yml deploy/Caddyfile deploy/oracle_setup.sh ubuntu@<VM-IP>:~/
ssh -i ~/.ssh/oracle_oncolens ubuntu@<VM-IP>
# on the VM:
bash oracle_setup.sh && mkdir -p ~/oncolens-deploy && mv docker-compose.yml Caddyfile ~/oncolens-deploy/
cd ~/oncolens-deploy && nano .env && docker compose up -d && docker compose logs -f api
```

**Tests:** `curl https://<API_DOMAIN>/health` returns model versions; the deployed frontend completes the e2e smoke test against production; the image built for arm64 starts on the VM; rollback works by pinning the previous `sha-` image tag.

**Gate 16:** public URL works end to end; deploy workflow green.

---

### Phase 17: Documentation, technical report & release

| Deliverable | Content |
|---|---|
| `README.md` | Hero GIF, live demo link, architecture diagram, results table with CIs, scaling chart, inference table, quickstart, limitations, disclaimer, dataset citations |
| `docs/research/REPORT.md` | Paper-style technical report: abstract, introduction, related work, methods (architectures, SSL, parallel strategies), experimental setup, results for RQ1–RQ4, discussion, limitations, ethics |
| `docs/BENCHMARKS.md` | Training scaling + inference tables and charts |
| `docs/MODEL_CARD.md`, `docs/DATA_CARD.md` | Intended use, out-of-scope use, data sources and licenses, metrics, subgroup results, caveats |
| `CITATION.cff`, `LICENSE` (MIT for code) | |
| Demo video (3–5 min) | Upload → result → report; then a quick tour of the scaling results |
| Release `v1.0.0` | Tag + GitHub release notes listing model versions and W&B run IDs |

**Gate 17 (project done):** every box in `docs/PROGRESS.md` is ticked, CI and deploy are green, and a fresh clone + README quickstart reproduces a test-set evaluation from the released models.

---

## Part E: Testing Strategy

### E.1 Test pyramid

| Level | Location | Runs where | Examples |
|---|---|---|---|
| Unit | `tests/unit/` | Local + CI (CPU) | storage, config, transforms, models, losses, metrics, schema, validator |
| Distributed (CPU) | `tests/distributed/` | Local + CI, `torchrun` 2 ranks on gloo | DDP gradient equivalence, distributed metrics, SimCLR gather, FSDP smoke |
| Integration | `tests/integration/` | Local | shards → loader → model → loss; pipeline image → Findings → PDF; API with TestClient |
| GPU | `tests/gpu/` (marker `gpu`) | Local laptop only | AMP numerics, `torch.compile`, ONNX CUDA/TensorRT parity |
| Multi-GPU | Kaggle runs | Kaggle 2× T4 | DDP vs single parity, FSDP vs DDP loss tracking, scaling benchmarks |
| End-to-end | `tests/e2e/` (marker `e2e`) | Local Docker + production URL | Playwright upload flow |

### E.2 Pytest markers & commands

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: long-running tests",
  "gpu: requires a CUDA GPU",
  "distributed: launches multiple processes",
  "e2e: end-to-end tests requiring running services",
  "data: requires datasets on ONCOLENS_ROOT",
]
addopts = "-q -m 'not gpu and not e2e and not data and not slow'"
```

```bash
pytest                                  # fast default suite (CI)
pytest -m distributed                   # multi-process CPU tests
pytest -m "gpu"                         # laptop GPU tests
pytest -m "data"                        # tests that need the datasets on G:
pytest -m "e2e"                         # needs docker compose up
```

### E.3 Testing rules for Claude Code
- Every new module ships with tests in the same commit.
- A failing test is never deleted or loosened to pass; fix the code or explain the change to the user and get agreement.
- Tests never read real secrets and never write outside a temporary directory or `$ONCOLENS_ROOT/tmp`.
- Data-dependent tests skip with a clear reason if G: isn't mounted.

---

## Part F: Configuration Reference

### F.1 Repository structure

```
oncolens/
├── CLAUDE.md
├── .claude/settings.json
├── .github/
│   ├── workflows/ {ci.yml, docker.yml, deploy-backend.yml, codeql.yml, healthcheck.yml}
│   ├── ISSUE_TEMPLATE/ {bug_report.yml, feature_request.yml}
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── dependabot.yml
│   └── CODEOWNERS
├── configs/
│   ├── config.yaml
│   ├── paths/      {local.yaml, kaggle.yaml, ci.yaml}
│   ├── data/       {pcam.yaml, isic2019.yaml, brain_mri.yaml, lc25000.yaml, nct_crc.yaml, ssl_pool.yaml, synthetic.yaml}
│   ├── model/      {simple_cnn.yaml, resnet18.yaml, resnet34.yaml, convnext_nano.yaml, convnext_tiny.yaml, vit_tiny.yaml, vit_small.yaml, vit_base.yaml}
│   ├── trainer/    {single.yaml, ddp.yaml, fsdp.yaml, deepspeed_z2.json}
│   ├── ssl/        {simclr.yaml, mae.yaml}
│   └── experiment/ {e001_m0_pcam_sanity.yaml, e002_m1_resnet18_pcam.yaml, ...}
├── src/oncolens/
│   ├── utils/      {storage.py, seed.py, io.py}
│   ├── data/       {datasets.py, shards.py, transforms.py, splits.py, synthetic.py}
│   ├── models/     {layers.py, simple_cnn.py, resnet.py, convnext.py, vit.py, heads.py, router.py, registry.py}
│   ├── ssl/        {simclr.py, mae.py, losses.py}
│   ├── baselines/  {timm_baseline.py}          # only place pretrained weights are allowed
│   ├── training/   {distributed.py, trainer.py, strategies.py, checkpoint.py, metrics.py, logging.py, ema.py}
│   ├── evaluation/ {metrics.py, calibration.py, bootstrap.py, subgroups.py}
│   ├── explain/    {gradcam.py, attention_rollout.py}
│   ├── inference/  {export_onnx.py, quantize.py, ort_session.py, batching.py}
│   ├── reporting/  {schema.py, templates/, narrator.py, validator.py, pdf.py}
│   ├── pipeline.py
│   └── api/        {main.py, routes.py, deps.py}
├── frontend/
├── scripts/        {check_env.py, download_data.sh, verify_raw.py, make_splits.py, check_leakage.py,
│                    build_shards.py, io_benchmark.py, train.py, train_deepspeed.py, evaluate.py,
│                    benchmark_scaling.py, benchmark_inference.py, export_onnx.py, fetch_results.py}
├── notebooks/kaggle/run_experiment.ipynb
├── setup/          {storage_setup.sh, wslconfig.txt}
├── deploy/         {Dockerfile, requirements-serve.txt, docker-compose.yml, Caddyfile, oracle_setup.sh}
├── data/           {splits/, manifests/}        # small, versioned; NO images here
├── tests/          {unit/, distributed/, integration/, gpu/, e2e/}
├── docs/           {PLAN.md, PROGRESS.md, BENCHMARKS.md, MODEL_CARD.md, DATA_CARD.md, research/, samples/, images/}
├── pyproject.toml
├── environment.yml
├── .pre-commit-config.yaml
├── .gitignore
├── .dockerignore
├── .env.example
├── LICENSE
├── CITATION.cff
└── README.md
```

### F.2 `environment.yml`

```yaml
name: oncolens
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - nodejs=20
```

PyTorch is installed separately with the command from pytorch.org (Phase 1.5), then `pip install -e ".[dev]"`.

### F.3 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "oncolens"
version = "0.1.0"
description = "From-scratch, distributed multi-cancer detection and reporting (research prototype)"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
  "numpy", "pandas", "scipy", "scikit-learn",
  "pillow", "opencv-python-headless", "albumentations", "imagehash",
  "webdataset", "pydicom",
  "hydra-core", "omegaconf", "wandb", "rich", "tqdm",
  "onnx", "onnxruntime",
  "fastapi", "uvicorn[standard]", "python-multipart", "pydantic>=2", "slowapi",
  "jinja2", "weasyprint", "matplotlib",
  "huggingface_hub",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "pytest-timeout", "httpx", "ruff", "mypy", "pre-commit", "dvc", "kaggle", "playwright"]
distributed = ["deepspeed"]          # install on Kaggle; optional locally
baselines = ["timm"]                 # reference baseline only
gpu-infer = ["onnxruntime-gpu"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: long-running tests",
  "gpu: requires a CUDA GPU",
  "distributed: launches multiple processes",
  "e2e: end-to-end tests requiring running services",
  "data: requires datasets on ONCOLENS_ROOT",
]
addopts = "-q -m 'not gpu and not e2e and not data and not slow'"
```

### F.4 Hydra configs

`configs/config.yaml`
```yaml
defaults:
  - paths: local
  - data: synthetic
  - model: simple_cnn
  - trainer: single
  - _self_

seed: 0
experiment_id: dev
hydra:
  run:
    dir: ${paths.runs}/hydra/${experiment_id}/${now:%Y-%m-%d_%H-%M-%S}
```

`configs/paths/local.yaml`
```yaml
root: ${oc.env:ONCOLENS_ROOT,/mnt/g/oncolens}
raw: ${paths.root}/raw
shards: ${paths.root}/shards
checkpoints: ${paths.root}/checkpoints
artifacts: ${paths.root}/artifacts
runs: ${paths.root}/runs
data_format: webdataset
require_mount: true
```

`configs/paths/kaggle.yaml`
```yaml
root: /kaggle/working
raw: /kaggle/input
shards: /kaggle/working/shards
checkpoints: /kaggle/working/checkpoints
artifacts: /kaggle/working/artifacts
runs: /kaggle/working/runs
data_format: folder          # Kaggle inputs are already extracted on fast storage
require_mount: false
```

`configs/paths/ci.yaml`
```yaml
root: ${oc.env:RUNNER_TEMP,/tmp}/oncolens
raw: ${paths.root}/raw
shards: ${paths.root}/shards
checkpoints: ${paths.root}/checkpoints
artifacts: ${paths.root}/artifacts
runs: ${paths.root}/runs
data_format: synthetic
require_mount: false
```

`configs/trainer/ddp.yaml`
```yaml
strategy: ddp
epochs: 30
batch_size_per_gpu: 128
grad_accum: 1
base_lr: 1.0e-3            # scaled by global_batch / 256
weight_decay: 0.05
warmup_epochs: 5
clip: 1.0
prefer_bf16: true          # fp16 automatically on T4
channels_last: true
compile: false
num_workers: 4
ema_decay: 0.9999
checkpoint_every: 1
```

Example experiment `configs/experiment/e002_m1_resnet18_pcam.yaml`
```yaml
# @package _global_
defaults:
  - override /data: pcam
  - override /model: resnet18
  - override /trainer: single
experiment_id: e002_m1_resnet18_pcam
model:
  init: scratch            # only "scratch" allowed outside baselines/
  num_classes: 2
trainer:
  epochs: 30
  batch_size_per_gpu: 256
tags: [RQ1, from_scratch, pcam]
```

### F.5 `src/oncolens/utils/storage.py` (behavior spec)

```python
def require_storage(paths_cfg) -> None:
    """Fail fast with a clear message when the external drive is missing."""
    # if not paths_cfg.require_mount: return
    # 1. ONCOLENS_ROOT exists and is a directory
    # 2. its parent mount (/mnt/g) is a mounted Windows drive (check /proc/mounts)
    # 3. write + delete a probe file in root/tmp
    # 4. warn if free space < 20 GB
    # On failure: raise StorageError("My Passport (G:) not mounted at /mnt/g. "
    #   "Plug it in, then run: sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g")
```

### F.6 `.gitignore`

```gitignore
# large files live on G:, never in git
*.pt
*.pth
*.ckpt
*.onnx
*.plan
*.engine
*.tar
*.zip
outputs/
multirun/
wandb/
checkpoints/
artifacts/
# python
__pycache__/
*.egg-info/
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.coverage
# secrets
.env
.env.local
kaggle.json
*.pem
*.key
# frontend
frontend/node_modules/
frontend/.next/
# OS
.DS_Store
Thumbs.db
```

### F.7 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9            # `pre-commit autoupdate` pins the latest
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: check-json
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-added-large-files
        args: [--maxkb=5000]
      - id: detect-private-key
```

### F.8 GitHub workflows

`.github/workflows/ci.yml`
```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install (CPU torch)
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check . && ruff format --check .
      - name: Types
        run: mypy src/oncolens
      - name: Unit + integration tests
        run: pytest --cov=oncolens
      - name: Distributed tests (2 CPU ranks, gloo)
        run: pytest -m distributed
      - name: Guard - no pretrained weights in research code
        run: |
          ! grep -rnE "pretrained\s*=\s*True|weights\s*=\s*['\"A-Z]" src/oncolens --include=*.py | grep -v "src/oncolens/baselines/"
  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci && npm run lint && npm run build
        if: hashFiles('frontend/package-lock.json') != ''
```

`.github/workflows/docker.yml`
```yaml
name: docker
on:
  push:
    branches: [main]
    tags: ["v*"]
permissions:
  contents: read
  packages: write
jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-qemu-action@v3
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: deploy/Dockerfile
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

`.github/workflows/deploy-backend.yml`
```yaml
name: deploy-backend
on:
  workflow_run:
    workflows: [docker]
    types: [completed]
    branches: [main]
  workflow_dispatch:
jobs:
  deploy:
    if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.ORACLE_HOST }}
          username: ${{ secrets.ORACLE_USER }}
          key: ${{ secrets.ORACLE_SSH_KEY }}
          script: |
            cd ~/oncolens-deploy
            docker compose pull
            docker compose up -d --remove-orphans
            docker image prune -f
            sleep 20 && curl -fsS http://localhost:8000/health
```

`.github/workflows/healthcheck.yml`
```yaml
name: healthcheck
on:
  schedule: [{ cron: "0 6 * * *" }]
  workflow_dispatch:
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS "https://${{ secrets.API_DOMAIN }}/health"
```

`.github/workflows/codeql.yml`
```yaml
name: codeql
on:
  push: { branches: [main] }
  schedule: [{ cron: "0 3 * * 1" }]
permissions:
  security-events: write
  contents: read
jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with: { languages: "python,javascript" }
      - uses: github/codeql-action/analyze@v3
```

`.github/dependabot.yml`
```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule: { interval: weekly }
  - package-ecosystem: github-actions
    directory: "/"
    schedule: { interval: weekly }
  - package-ecosystem: docker
    directory: "/deploy"
    schedule: { interval: weekly }
  - package-ecosystem: npm
    directory: "/frontend"
    schedule: { interval: weekly }
```

`.github/PULL_REQUEST_TEMPLATE.md`
```markdown
### What changed
### Why
### Tests
- [ ] `pytest` and `pytest -m distributed` pass
- [ ] New code has tests
- [ ] For experiments: LOG.md entry + W&B link
### Checklist
- [ ] No data, weights, or secrets committed
- [ ] No pretrained weights outside `baselines/`
```

GitHub repository secrets (set with `gh secret set NAME`, which prompts for the value): `HF_TOKEN`, `ORACLE_HOST`, `ORACLE_USER`, `ORACLE_SSH_KEY`, `API_DOMAIN`.

### F.9 Deployment files

`deploy/Dockerfile`
```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libpango-1.0-0 libpangoft2-1.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/requirements-serve.txt .
RUN pip install -r requirements-serve.txt          # onnxruntime (CPU), fastapi, weasyprint... no torch
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .
RUN useradd -m appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "oncolens.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

Inference code paths used by the API must import only ONNX Runtime and NumPy (no PyTorch), so the serving image stays small.

`deploy/docker-compose.yml`
```yaml
services:
  api:
    image: ghcr.io/<GITHUB_USER>/oncolens:main
    restart: unless-stopped
    env_file: .env
    environment:
      - MODEL_REPO=<HF_USER>/oncolens-models
      - ALLOWED_ORIGINS=https://oncolens.vercel.app
      - LLM_URL=http://llm:8080
    volumes:
      - models:/home/appuser/.cache/huggingface
    expose: ["8000"]
  llm:                                    # optional narrator; verify the current llama.cpp server image name
    image: ghcr.io/ggml-org/llama.cpp:server
    restart: unless-stopped
    command: ["-m", "/models/narrator-q4_k_m.gguf", "--host", "0.0.0.0", "--port", "8080", "-c", "2048", "-t", "2"]
    volumes:
      - ./llm-models:/models
    expose: ["8080"]
  caddy:
    image: caddy:2
    restart: unless-stopped
    env_file: .env
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
volumes:
  models:
  caddy_data:
```

`deploy/Caddyfile`
```caddy
{$API_DOMAIN} {
    encode gzip
    request_body {
        max_size 10MB
    }
    reverse_proxy api:8000
    header {
        Strict-Transport-Security "max-age=31536000"
        X-Content-Type-Options nosniff
        Referrer-Policy no-referrer
    }
}
```

`deploy/oracle_setup.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
sudo apt update && sudo apt -y upgrade
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# Oracle Ubuntu images ship restrictive iptables rules: open 80/443
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
echo "Log out and back in (docker group), then: cd ~/oncolens-deploy && docker compose up -d"
```

Frontend on Vercel:
```bash
# ▶ USER RUNS
cd frontend && npm i -g vercel && vercel login && vercel link
vercel env add NEXT_PUBLIC_API_URL production     # https://<VM-IP>.sslip.io
vercel --prod
```

### F.10 `.wslconfig`

Saved in the repo as `setup/wslconfig.txt`, copied to `C:\Users\<you>\.wslconfig` (Phase 1.3):
```ini
[wsl2]
memory=24GB
swap=8GB
```

### F.11 `setup/storage_setup.sh`

```bash
#!/usr/bin/env bash
# Usage: bash setup/storage_setup.sh /mnt/g/oncolens
set -euo pipefail
ROOT="${1:-/mnt/g/oncolens}"
DRIVE="$(dirname "$ROOT")"

if ! grep -qs " $DRIVE " /proc/mounts; then
  echo "ERROR: $DRIVE is not mounted. Plug in My Passport (G:), then run:"
  echo "  sudo mkdir -p $DRIVE && sudo mount -t drvfs G: $DRIVE"
  exit 1
fi

mkdir -p "$ROOT"/{raw,shards,checkpoints,artifacts,runs/hydra,runs/wandb,runs/kaggle,cache/hf,cache/torch,cache/pip,dvc-remote,tmp}

echo "Write-speed test (1 GB) on $ROOT ..."
dd if=/dev/zero of="$ROOT/tmp/speedtest.bin" bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
rm -f "$ROOT/tmp/speedtest.bin"

MARK_START="# >>> oncolens storage >>>"
MARK_END="# <<< oncolens storage <<<"
if ! grep -q "$MARK_START" ~/.bashrc; then
  cat >> ~/.bashrc <<EOF
$MARK_START
export ONCOLENS_ROOT="$ROOT"
export HF_HUB_CACHE="\$ONCOLENS_ROOT/cache/hf"
export TORCH_HOME="\$ONCOLENS_ROOT/cache/torch"
export PIP_CACHE_DIR="\$ONCOLENS_ROOT/cache/pip"
export WANDB_DIR="\$ONCOLENS_ROOT/runs/wandb"
$MARK_END
EOF
  echo "Added environment variables to ~/.bashrc"
fi
df -h "$ROOT" | tail -1
echo "Done. Run: source ~/.bashrc"
```

### F.12 `.env.example`

```bash
# Names only. Real values go in .env (never committed, never read by Claude Code).
ONCOLENS_ROOT=/mnt/g/oncolens
HF_TOKEN=
API_DOMAIN=
MODEL_REPO=<HF_USER>/oncolens-models
ALLOWED_ORIGINS=https://oncolens.vercel.app
```

---

## Part G: Datasets Reference

Dataset identifiers are given from prior knowledge. **In Phase 3, Claude Code verifies each link, the current license, and access requirements before anything is downloaded.**

| Key | Dataset | Approx. size | Source | License (verify) | Use |
|---|---|---|---|---|---|
| `isic2019` | ISIC 2019 Challenge (≈25k dermoscopy images, 8 classes) | ~9 GB | Kaggle mirror / ISIC Archive | CC-BY-NC | Module A |
| `ham10000` | HAM10000 | ~3 GB | Harvard Dataverse / Kaggle | CC-BY-NC | Module A external test (de-duplicated against ISIC 2019) |
| `brain_mri` | Brain Tumor MRI Dataset (≈7k images, 4 classes) | ~150 MB | Kaggle | see page | Module B |
| `lc25000` | Lung and Colon Cancer Histopathological Images (25k) | ~1.8 GB | Kaggle | see page | Module C |
| `nct_crc` | NCT-CRC-HE-100K + CRC-VAL-HE-7K | ~12 GB | Zenodo | CC-BY 4.0 | Module C training/external |
| `pcam` | Histopathologic Cancer Detection (PCam-derived, ≈220k patches) | ~7 GB | Kaggle competition | competition rules | Module D; main dataset for parallel-training benchmarks |

Total core storage on G: (archives + shards + checkpoints): **≈ 120–180 GB**. Stretch datasets (BraTS, LIDC-IDRI/LUNA16, TCGA) are out of scope for v1.

Known caveat: LC25000 contains augmentation-derived near-duplicates, so its accuracy is inflated; the leakage check (Phase 4.2) and external validation on NCT-CRC data are mandatory and are discussed in the report.

---

## Part H: Timeline & Compute Budget

### H.1 Timeline (guided pace, ~6–10 hours/day)

From-scratch research takes longer than fine-tuning pretrained models; a realistic schedule is **4 weeks** (compressible to ~3).

| Week | Phases | Outcome |
|---|---|---|
| 1 | 0–5, start 6 | Environment, storage on G:, repo + CI, data on G:, splits, shards, trainer with all tests green; first ResNet run |
| 2 | 6–9 | ResNet, ConvNeXt, ViT from scratch; first Kaggle DDP; FSDP/ZeRO-2; scaling benchmarks |
| 3 | 10–11 | SimCLR + MAE from scratch; low-label studies; full evaluation |
| 4 | 12–17 | Router, explainability, ONNX/TensorRT, reports, API/UI, deployment, technical report, release |

**Use the laptop for long single-GPU runs overnight** (unlimited hours) and **Kaggle for anything that needs 2 GPUs.** Kaggle quota resets weekly, so plan multi-GPU runs across weeks 2–3.

### H.2 Compute budget (rough estimates; replace with measured numbers after Phase 7)

| Workload | Where | Est. GPU-hours |
|---|---|---|
| Scaling benchmarks (DDP, FSDP, ZeRO-2) | Kaggle 2× T4 | 3–5 |
| Main DDP trainings (PCam, ISIC; M1–M3) | Kaggle 2× T4 | 10–15 |
| SimCLR batch study + MAE | Kaggle 2× T4 | 10–15 |
| Seeds, small datasets, fine-tuning, ablations | Laptop | 40–80 (overnight) |

Kaggle total ≈ 25–35 hours spread over 2–3 weekly quotas. Running 2× T4 may consume quota faster than wall-clock time; check the quota display before each long run.

---

## Part I: Risks, Licensing & Ethics

### I.1 Risks & mitigations

| Risk | Mitigation |
|---|---|
| My Passport disconnects or sleeps mid-run | USB selective suspend off; atomic checkpoints every epoch; `require_storage()` fails fast; resume from last checkpoint |
| USB HDD too slow | Zipped archives → sequential shards; 24 GB RAM page cache; I/O benchmark gate; optional ext4 VHDX (Phase 4.6) |
| Drive failure / data loss | Raw data can be re-downloaded; best checkpoints mirrored to a private HF repo; code on GitHub |
| Kaggle session killed or quota exhausted | Per-epoch checkpoints pushed to HF; resumable training; short benchmarks before long runs |
| 8 GB VRAM limits on laptop | bf16, gradient checkpointing, gradient accumulation |
| From-scratch models underperform | Expected and itself a result (RQ1); SSL (RQ2) addresses it; report honestly |
| Data leakage | Patient-level splits, perceptual-hash duplicate checks, external test sets, test set used once |
| LLM hallucination in reports | LLM only paraphrases JSON; validator + template fallback |
| Free tiers change | Portable Docker; Plan B static browser inference |
| Overclaiming clinical value | Disclaimers everywhere, limitations, abstention |

### I.2 Licensing & ethics
- Code: MIT. Each dataset keeps its own license; several are **non-commercial** (CC-BY-NC), so the project and demo are non-commercial. Raw datasets are never redistributed; only redistributable sample images appear in the repo/demo, with attribution.
- Trained weights inherit the most restrictive data license; stated in the model card.
- Uploaded images are processed in memory and not stored. No analytics on uploaded content.
- Clinical use is explicitly out of scope.
- All datasets are cited in `README.md` and `CITATION.cff`.

---

## Appendix 1: CLAUDE.md

Claude Code creates this file in the repo root in Phase 0.

~~~markdown
# CLAUDE.md: OncoLens

## Project
From-scratch, distributed multi-cancer detection research project. Full spec: docs/PLAN.md.
Progress tracker: docs/PROGRESS.md. Experiments: docs/research/EXPERIMENTS.md. Lab notebook: docs/research/LOG.md.

## Session start
1. Read this file and docs/PROGRESS.md (and the relevant phase in docs/PLAN.md).
2. Tell the user the current phase/step, what's next, and any open issues. Wait for "go".

## Your role
Research guide + pair programmer. The user is learning and showcasing model building and parallel GPU training.
Explain concepts briefly and precisely before implementing them. Keep the user in control.

## Step protocol (every step)
1. Goal (1–2 sentences) and why it matters.
2. Concept: short explanation; name the original paper if relevant.
3. Changes: list files, then make them.
4. Commands for the user in a code block labeled "▶ USER RUNS", with expected output.
5. Wait for output; diagnose errors before continuing.
6. Run allowed tests; hand over the rest.
7. Tick docs/PROGRESS.md, add a LOG.md entry for experiments, make a small commit.

## Execution modes
- GUIDED (default): you may run pytest, ruff, mypy, git status/diff/add/commit, ls, reading source files, short python -c checks, nvidia-smi.
  Hand to the user: downloads, shard building, training/benchmarks beyond pytest smoke tests, docker, sudo, ssh/scp, git push, deployment, Windows/PowerShell steps, Kaggle steps, logins.
- AUTO: only after the user says "switch to AUTO mode". Then you may also run downloads, shard building, and local training, asking before each.
  Still always the user's job: sudo, Windows/PowerShell, logins, ssh/scp, git push, docker, deployment, Kaggle.

## Machine & storage
- WSL2 Ubuntu on Windows. GPU: RTX 4070 Laptop, 8 GB VRAM, bf16. RAM 32 GB (WSL: 24 GB).
- Repo: ~/projects/oncolens (internal SSD). Conda env: oncolens.
- ALL large files live under $ONCOLENS_ROOT = /mnt/g/oncolens (My Passport external USB drive, G:).
  Datasets, shards, checkpoints, caches, runs, exports: never in the repo or home dir. Use configs/paths/*.yaml.
- Every data/checkpoint script calls oncolens.utils.storage.require_storage() first.
- HDD rules: keep archives zipped; stream into WebDataset shards; never extract many small files on G:; atomic checkpoint writes.
- If G: is missing: tell the user to plug it in and run `sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g`.
- Kaggle: 2× T4 (fp16), data in /kaggle/input, outputs in /kaggle/working (~20 GB), secrets via Kaggle Secrets. Use paths=kaggle.

## Hard rules
- Phase gates: never start a phase until the previous gate passes.
- From scratch only: no pretrained weights (no pretrained=True, no torchvision/timm weights) outside src/oncolens/baselines/.
- Models are hand-written in plain PyTorch (src/oncolens/models, src/oncolens/ssl).
- Every new module ships with tests in the same commit. Never delete or weaken a failing test to make it pass.
- Secrets: never read/print .env, ~/.kaggle, ~/.ssh, ~/.netrc, or tokens. Never ask the user to paste a key.
- Check torch version before using FSDP / distributed checkpoint / torch.compile APIs.
- Report results exactly as measured, including negative results. Never tune on the test set.
- Conventional commits: feat(...), fix(...), test(...), docs(...), exp(...).

## Common commands
- Fast tests: `pytest`
- Distributed CPU tests: `pytest -m distributed`
- Lint/types: `ruff check . && ruff format --check . && mypy src/oncolens`
- Local train (user runs): `python scripts/train.py experiment=<ID>`
- Local DDP smoke (CPU): `CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=2 scripts/train.py experiment=<ID> paths=ci data=synthetic trainer=ddp trainer.epochs=1`
- Kaggle (user runs in notebook): `torchrun --standalone --nproc_per_node=2 scripts/train.py experiment=<ID> paths=kaggle`
~~~

---

## Appendix 2: .claude/settings.json

Claude Code creates this in Phase 0. Deny rules block Claude Code's own access; ask rules force a confirmation prompt. (Deny rules on reads protect Claude's file tools; CLAUDE.md additionally forbids reading secrets through shell commands.)

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(pytest)",
      "Bash(pytest *)",
      "Bash(ruff *)",
      "Bash(mypy *)",
      "Bash(git status)",
      "Bash(git diff)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(nvidia-smi)"
    ],
    "ask": [
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(pip install *)",
      "Bash(python scripts/*)"
    ],
    "deny": [
      "Read(./.env)",
      "Read(./.env.local)",
      "Read(~/.kaggle/**)",
      "Read(~/.ssh/**)",
      "Read(~/.netrc)",
      "Read(~/.cache/huggingface/token)",
      "Bash(sudo *)",
      "Bash(ssh *)",
      "Bash(scp *)",
      "Bash(git push *)",
      "Bash(docker *)",
      "Bash(kaggle *)",
      "Bash(vercel *)",
      "Bash(gh secret *)"
    ]
  }
}
```

If Claude Code reports a settings error at startup, run `/permissions` inside Claude Code to review and fix the rules. In AUTO mode the user may move specific entries (e.g. `Bash(kaggle *)`) from `deny` to `ask`.

---

## Appendix 3: docs/PROGRESS.md

~~~markdown
# OncoLens Progress
Current phase: 0 | Mode: GUIDED | Last updated: <date>

## Phase 0: Workspace
- [ ] 0.1 CLAUDE.md  - [ ] 0.2 settings.json  - [ ] 0.3 PROGRESS.md  - [ ] 0.4 research docs  - [ ] 0.5 .gitignore/.env.example  - [ ] 0.6 commit
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
- (none)
~~~

---

## Appendix 4: docs/research/EXPERIMENTS.md & LOG.md

`docs/research/EXPERIMENTS.md`
~~~markdown
# Experiment Registry
Rules: register an experiment BEFORE running it. Fix the metric and the comparison in advance.
Primary metric: macro AUROC on validation (model selection) and test (final, once).
Seeds: 0, 1, 2 for any result reported in REPORT.md.

| ID | RQ | Dataset | Model | Init | Strategy | GPUs | Precision | Global batch | Epochs | Seeds | Status | W&B group | Result |
|----|----|---------|-------|------|----------|------|-----------|--------------|--------|-------|--------|-----------|--------|
| e001 | - | pcam | M0 SimpleCNN | scratch | single | 4070 | bf16 | 256 | 3 | 0 | planned | sanity | |
| e002 | RQ1 | pcam | M1 ResNet-18 | scratch | single | 4070 | bf16 | 256 | 30 | 0,1,2 | planned | rq1-pcam | |
| e010 | RQ1/RQ3 | pcam | M1 ResNet-18 | scratch | ddp | 2×T4 | fp16 | 512 | 30 | 0,1 | planned | rq3-scaling | |
~~~

`docs/research/LOG.md` entry template
~~~markdown
## <YYYY-MM-DD> <experiment ID>: <short title>
- Hypothesis:
- Config / command:
- Hardware & strategy: (e.g. Kaggle 2×T4, DDP, fp16, global batch 512)
- Result: (metric ± std, throughput, peak VRAM)
- W&B: <link>
- Observations:
- Decision / next step:
~~~
