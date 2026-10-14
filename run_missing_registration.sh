#!/bin/bash
# ROICaT for the five last-learning -> first-reversal transitions that the consecutive
# pair list missed, plus the learned-reversal alternative where one exists.
# Takes the reversal-crossing cohort from 9 mice to as many as 14.
set -u
ROICAT_PY="$HOME/.venvs/roicat/bin/python"
[ -x "$ROICAT_PY" ] || { echo "ROICaT venv missing at $ROICAT_PY"; exit 1; }
cd "$(dirname "$0")"
PAIRS="/Volumes/hard_drive/viral/nlgf_derived/results/missing_transition_pairs.csv"
[ -f "$PAIRS" ] || { echo "no pair list at $PAIRS"; exit 1; }
LOCK=/tmp/viral_registration.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "registration already running (pid $(cat "$LOCK"))."; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "missing-transition registration started $(date)"
"$ROICAT_PY" -m viral.nlgf.register_roicat --pairs-csv "$PAIRS" --um-per-pixel 1.0 2>&1 \
  | grep -vE "it/s\]|MB/s\]|^\{'general'"
echo "finished $(date)"
