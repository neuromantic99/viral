#!/bin/bash
# Cross-session ROI registration with ROICaT.
#
#   ./run_registration.sh                 # all consecutive session pairs
#   ./run_registration.sh transition      # just last-learning -> first-reversal
#   ./run_registration.sh all             # every pair within a mouse (slow)
#
# Resumable: pairs whose match table already exists are skipped, so if this is
# interrupted, just run it again. Progress goes to registration.log.
#
# Outputs, all under /Volumes/hard_drive/viral/nlgf_derived/
#   results/matches_<mouse>_<dateA>_<dateB>.csv   matched cells, as spks row indices
#   results/registration_diagnostics.csv          match rate + silhouette per pair
#   roicat_qc/qc_<mouse>_<dateA>_<dateB>.png      ROIs each day, matches highlighted
set -u
MODE="${1:-consecutive}"
# ROICaT lives in its own venv, NOT the project one: it pulls torch, jupyter and ~150
# other packages and wants a newer numpy than the project pins. Kept outside /tmp so it
# survives a reboot.
ROICAT_PY="$HOME/.venvs/roicat/bin/python"
if [ ! -x "$ROICAT_PY" ]; then
  echo "ROICaT venv missing. Recreate with:"
  echo "  /opt/homebrew/bin/python3.10 -m venv ~/.venvs/roicat"
  echo "  ~/.venvs/roicat/bin/pip install 'roicat[all]' statsmodels"
  exit 1
fi
cd "$(dirname "$0")"
LOCK=/tmp/viral_registration.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "registration already running (pid $(cat "$LOCK")). Two copies thrash the drive."
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "mode=$MODE  started $(date)"
"$ROICAT_PY" -m viral.nlgf.register_roicat --mode "$MODE" --um-per-pixel 1.0 2>&1 \
  | grep -vE "it/s\]|MB/s\]|^\{'general'"
echo "finished $(date)"
