# Stage 3 Drawer, attempt 1: four seeds in one folder

**Kept as evidence. Not used by any later stage.**

- **What ran:** `run.py E3` at commit b621e00 started the Stow run and the four Drawer seeds (0–3) together, at 14:26:55 on 2026-11-11.
- **What went wrong:** `scripts/drawer_stage3_train.py` named its run folder by the start time in whole seconds, and reused a folder that already existed. All four seeds started in the same second, so they all wrote into `1794407215/`.
  - `config.json` is seed 1's.
  - `summary.json` is seed 3's, the last to finish.
  - `sac_hole_best.zip` and `sac_hole_final.zip` are whichever seed saved last.
  - Each seed's "best checkpoint" test result in its log may come from another seed's file.
- **What is still sound:** each seed's log in `logs/drawer_train_seed<k>.attempt1_collided.log`, up to the test results. These are the validation curves and the final-model test results; the final model was evaluated from memory.
- **Fix:** the folder is now named `seed<k>_<start time>`, and a run refuses to write into an existing folder. The four seeds were then retrained from scratch at the fixed commit, into `E3_stage3/drawer/`.
- **Renamed, not changed:** this folder (from `E3_stage3/drawer/`), the four seeds' logs and records, the checkpoint manifest step's log and record, and the runner's console output (`logs/run_E3.attempt1.out`).
