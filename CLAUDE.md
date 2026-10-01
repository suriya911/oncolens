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
7. Tick docs/PROGRESS.md, update the README roadmap status, add a LOG.md entry for experiments, make a small commit.
8. At the end of each phase (module): push to GitHub (`git push`), the user has asked for this.

## Execution modes
- GUIDED (default): you may run pytest, ruff, mypy, git status/diff/add/commit/push, ls, reading source files, short python -c checks, nvidia-smi.
  Hand to the user: downloads, shard building, training/benchmarks beyond pytest smoke tests, docker, sudo, ssh/scp, deployment, Windows/PowerShell steps, Kaggle steps, logins.
- AUTO: only after the user says "switch to AUTO mode". Then you may also run downloads, shard building, and local training, asking before each.
  Still always the user's job: sudo, Windows/PowerShell, logins, ssh/scp, docker, deployment, Kaggle.

## Machine & storage (deviation from PLAN: repo lives on G:)
- WSL2 Ubuntu on Windows. GPU: RTX 4070 Laptop, 8 GB VRAM, bf16. RAM 32 GB (WSL: 24 GB).
- Repo work tree: /mnt/g/OncoLens (My Passport, G:). Git metadata: ~/git/oncolens.git (SSD), because
  the drvfs mount does not allow chmod, which git needs inside .git/. `.git` in the repo is a gitdir pointer file.
- Windows paths are case-insensitive: /mnt/g/oncolens IS /mnt/g/OncoLens. So the data root is NOT the plan's
  /mnt/g/oncolens. Data root: $ONCOLENS_ROOT = /mnt/g/oncolens-data.
- ALL large files live under $ONCOLENS_ROOT. Datasets, shards, checkpoints, caches, runs, exports: never in the repo or home dir. Use configs/paths/*.yaml.
- Conda env: oncolens (on SSD, ~/miniforge3).
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
- Conventional commits: feat(...), fix(...), test(...), docs(...), exp(...). No Claude co-author or "Generated with Claude" lines in commits or PRs.

## Common commands
- Fast tests: `pytest`
- Distributed CPU tests: `pytest -m distributed`
- Lint/types: `ruff check . && ruff format --check . && mypy src/oncolens`
- Local train (user runs): `python scripts/train.py experiment=<ID>`
- Local DDP smoke (CPU): `CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=2 scripts/train.py experiment=<ID> paths=ci data=synthetic trainer=ddp trainer.epochs=1`
- Kaggle (user runs in notebook): `torchrun --standalone --nproc_per_node=2 scripts/train.py experiment=<ID> paths=kaggle`
