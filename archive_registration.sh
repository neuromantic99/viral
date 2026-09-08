#!/bin/bash
# Archive ROICaT's intermediate output to the hard drive, after registration finishes.
#
#   ./archive_registration.sh          # archive, keep local copies
#   ./archive_registration.sh --prune  # archive, then delete the local copies
#
# During a run, ROICaT's bulk output (run_data.richfile: the ROInet embeddings and
# similarity graphs, ~105 MB of thousands of small files per pair) is written to
# ~/.cache/viral_roicat_out on local disk. Writing that many small files directly to
# the exFAT drive took 80 minutes per pair, because each one spawns an AppleDouble
# companion and the drive is slow at metadata operations.
#
# One tar per pair copies fast, because it is a single large sequential write. The
# point of keeping it at all: run_data holds the expensive part - the ~10 minutes of
# CPU per pair - so re-clustering at a different threshold later would not need the
# embeddings recomputed.
set -u
SRC="$HOME/.cache/viral_roicat_out"
DST="/Volumes/hard_drive/viral/nlgf_derived/roicat_archive"
PRUNE=0
[ "${1:-}" = "--prune" ] && PRUNE=1

[ -d "$SRC" ] || { echo "nothing to archive: $SRC does not exist"; exit 0; }
mkdir -p "$DST" || { echo "cannot write to $DST - is the drive mounted?"; exit 1; }

if pgrep -f register_roicat > /dev/null; then
  echo "registration is still running - wait for it to finish first"; exit 1
fi

n_done=0; n_skip=0; n_fail=0
for dir in "$SRC"/*/; do
  [ -d "$dir" ] || continue
  pair=$(basename "$dir")
  out="$DST/$pair.tar.gz"
  if [ -f "$out" ]; then
    echo "  skip    $pair (already archived)"; n_skip=$((n_skip+1)); continue
  fi
  printf "  archive %s ... " "$pair"
  # write to a .part file so an interrupted copy is never mistaken for a good one
  if tar -czf "$out.part" -C "$SRC" "$pair" 2>/dev/null && tar -tzf "$out.part" >/dev/null 2>&1; then
    mv "$out.part" "$out"
    size=$(du -h "$out" | cut -f1)
    echo "ok ($size)"
    n_done=$((n_done+1))
    [ "$PRUNE" = "1" ] && rm -rf "$dir"
  else
    rm -f "$out.part"; echo "FAILED"; n_fail=$((n_fail+1))
  fi
done

echo
echo "archived $n_done, skipped $n_skip, failed $n_fail"
[ "$PRUNE" = "1" ] && echo "local copies removed" || \
  echo "local copies kept in $SRC (rerun with --prune to remove)"
du -sh "$DST" 2>/dev/null | awk '{print "archive size:", $1}'
