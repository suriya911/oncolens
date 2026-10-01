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

## 2026-10-01 setup: G: drive write speed
- Command: `bash setup/storage_setup.sh /mnt/g/oncolens-data`
- Result: 1 GiB sequential write in 21.4 s = 50.2 MB/s (below the 80-150 MB/s typical for USB HDDs).
- Observations: drvfs (9p) overhead on WSL likely costs throughput. Phase 4.5 I/O benchmark decides if the ext4 VHDX (Step 4.6) is needed.
