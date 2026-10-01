# Lab Notebook

Entry template:

```markdown
## <YYYY-MM-DD> <experiment ID>: <short title>
- Hypothesis:
- Config / command:
- Hardware & strategy: (e.g. Kaggle 2×T4, DDP, fp16, global batch 512)
- Result: (metric ± std, throughput, peak VRAM)
- W&B: <link>
- Observations:
- Decision / next step:
```

---

## 2026-09-30 setup: workspace decisions
- Repo work tree kept on G: at /mnt/g/OncoLens (user choice). Git metadata on SSD at ~/git/oncolens.git,
  because the drvfs mount rejects chmod (`chmod on .git/config.lock failed: Operation not permitted`).
- Windows paths are case-insensitive, so the plan's data root /mnt/g/oncolens collides with the repo.
  Data root moved to /mnt/g/oncolens-data.
