#!/bin/bash
# ROICaT registration for the Oligo-BACE1-KO consecutive session pairs.
#
#   ./run_oligo_registration.sh
#
# Same machinery as run_registration.sh, but driven from an explicit pair list
# (results/oligo_pairs_to_register.csv, written by viral/nlgf/oligo.py) rather than
# from --mode, because the Oligo mice are not in the default WT/NLGF cohort.
#
# Shares the SAME lockfile as run_registration.sh: two ROICaT runs thrash the drive
# and we have lost a night to that once already.
set -u
ROICAT_PY="$HOME/.venvs/roicat/bin/python"
if [ ! -x "$ROICAT_PY" ]; then
  echo "ROICaT venv missing at $ROICAT_PY"; exit 1
fi
cd "$(dirname "$0")"
PAIRS="/Volumes/hard_drive/viral/nlgf_derived/results/oligo_pairs_to_register.csv"
if [ ! -f "$PAIRS" ]; then
  echo "no pair list at $PAIRS; run: python -m viral.nlgf.oligo"; exit 1
fi
LOCK=/tmp/viral_registration.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "registration already running (pid $(cat "$LOCK"))."; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "oligo registration started $(date)"
"$ROICAT_PY" -m viral.nlgf.register_roicat --pairs-csv "$PAIRS" --um-per-pixel 1.0 2>&1 \
  | grep -vE "it/s\]|MB/s\]|^\{'general'"
echo "finished $(date)"
