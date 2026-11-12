# Direct runs, attempt 1: stopped, not results

On 2026-11-12 the runner started nine training processes at once (Stow, four targeted and these
four direct runs). Together they used all of the laptop's memory (13 of 14 GB, and all 4 GB of
swap) and froze it. At about 15,000 of their 30,000 steps, these four runs were stopped
(SIGTERM, exit code -15) to free memory. The Stow and targeted runs finished normally.

These folders hold what the stopped runs had written: their configurations and the checkpoints
saved so far. They are kept as a record and are not used. The four runs were rerun from scratch,
with the same seeds and settings, once the other five had finished: `../direct/`.

Their records and logs are kept as `records/direct_train_seed<k>.attempt1_stopped.json` and
`logs/direct_train_seed<k>.attempt1_stopped.log`; the runner's console output of this attempt is
`logs/run_R1.attempt1.out`.
