# viral

## Running code

- Use **`/Users/jamesrowland/Code/viral/venv/bin/python`** (3.12.7, pandas 2.2.3, pydantic 2.10.6)
  for everything. `venv/` is gitignored and there is no pyproject/lockfile. Every other
  interpreter on the machine fails in a different way: homebrew pythons lack pandas or
  statsmodels, anaconda has pydantic v1 (`viral.models` won't import), and
  `grosmark_analysis.py` uses a nested same-quote f-string that needs Python >= 3.12.
- Exception: `viral.nlgf.register_roicat` is run with `~/.venvs/roicat/bin/python`.
- Set `MPLBACKEND=Agg` for headless figure scripts. Bare `python` is not on PATH in a
  non-interactive shell.

## Working files

- Never put logs, scripts, intermediate tables or venvs in `/tmp` (including the session
  scratchpad under `/private/tmp/claude-501/...`). macOS clears it on restart and this has
  lost work twice. Use **`~/.cache/viral_scratch/`**.
- Analysis outputs go on the data drive under `nlgf_derived/results/`. The drive unmounts
  occasionally, so long jobs should keep their progress marker in `~/.cache/viral_scratch/`
  and check the drive is mounted before writing.

## Experimental setup

- The VR corridor is **visual**: the three landmarks at 45/90/135 cm and the two
  reward-mapped textures are visual cues, not tactile. The docstrings in
  `viral/nlgf/style.py` (`landmark_lines`) and `viral/nlgf/fig_tier4.py` saying "tactile"
  are wrong. The error came from Chang et al. (2020), whose rig uses a physical belt. Treat
  any Chang-derived description in the repo as being about their rig, not ours.
- The **Oligo-BACE1-KO** mice (JB014, JB015, JB018, JB020, JB022) are on an **App-NL-G-F
  background**, even though `viral/utils.py:get_genotype` returns it as a flat label next to
  "NLGF" and "WT". That arm is a rescue experiment: the primary contrast is Oligo vs
  **NLGF**, never Oligo vs WT. A rescue looks like NLGF impaired and Oligo ≈ WT. A measure
  where NLGF and WT don't differ has nothing to rescue.
- **Recording-era confound:** the Oligo mice were recorded Oct-Dec 2024 and WT from
  2024-12-10 onwards, so Oligo-vs-WT is confounded with recording era, and no
  contemporaneous WTs exist. NLGF spans both eras, so the batch effect can be bounded:
  `viral/nlgf/era.py:confound_scale()` divides a genotype difference by the NLGF early-vs-late
  effect on the same measure. A ratio below 1 is no evidence. Oligo vs NLGF is time-matched.

## Statistics

- With ~7-8 mice per group, use leave-pair-out AUC, not pooled leave-one-out. Pooled LOO
  gives spurious below-chance AUCs at this n.
- The gap confound on cross-day drift is already handled in `viral/nlgf/fig_stability.py`.
