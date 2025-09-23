"""
Utils to fix broken tiff files by saving all but the last, presumably broken, frame.
Recommended to fix in this order:
1. If you set up suite2p to be verbose (print the tiff file name when loading), you can skip step 2.
2. Use "find_broken_tiffs_in_session" to get a list of broken tiff files in a given session directory.
3. Use "save_tiff_until_broken" on each broken tiff file to save a new tiff file with all but the last frame.
4. Carefully move the broken file from the original directory to the "damaged_tiffs" directory.
5. Now re-run suite2p on the session directory.
"""

import tifffile
import sys
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple
from ScanImageTiffReader import ScanImageTiffReader

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import TEMP_CACHE_PATH, TIFF_UMBRELLA
from viral.cache_2p_sessions import extract_metadata, check_no_dropped_frames


def extract_metadata_broken_tiff(
    tiff_path: Path,
) -> Tuple[int, np.ndarray, List[float]]:
    """Extract metadata from tiff file until the broken frame (last frame)"""
    with tifffile.TiffFile(tiff_path) as tiff:
        n_frames = len(tiff.pages)
        timestamps = list()
        for idx in range(n_frames - 1):
            # read all but the last, presumably broken, frame
            description = tiff.pages[idx].tags["ImageDescription"].value
            timestamp_match = re.search(
                r"frameTimestamps_sec\s*=\s*(-?\d+\.\d+)", description
            )
            timestamps.append(float(timestamp_match[1]))

        if timestamps is None or len(timestamps) != n_frames - 1:
            raise ValueError("Could not extract all timestamps from tiff description")
        # epoch (same across frames, grab from first description)
        description0 = tiff.pages[0].tags["ImageDescription"].value
        epoch_match = re.search(r"epoch\s*=\s*\[([^\]]+)\]", description0)
        epoch = list(map(float, epoch_match[1].split()))
        if epoch is None:
            raise ValueError("Could not extract epoch from tiff description")
        return n_frames - 1, epoch, timestamps


def save_tiff_until_broken(tiff_path: Path) -> None:
    """Save broken tiff file until the broken frame (last frame)."""
    print(f"Fixing {tiff_path}")
    # need to read the entire file to get the number of frames
    with tifffile.TiffFile(tiff_path) as tiff:
        n_frames = len(tiff.pages)
    # then read all but the last, presumably broken, frame
    tiff_read = tifffile.imread(tiff_path, key=range(0, n_frames - 1, 1))
    output_path = tiff_path.parent / (tiff_path.stem + "_until_broken.tiff")
    tifffile.imwrite(output_path, tiff_read, bigtiff=True)
    print("Done")


def check_if_tiff_is_broken(tiff_path: Path) -> bool:
    """Checks whether a tiff file is broken."""
    # Doing this with ScanImageTiffReader first as it is faster than tifffile
    print(f"Checking {tiff_path}")
    try:
        tiff_file = ScanImageTiffReader(str(tiff_path))
        # tiff_file.data()
    except Exception as e:
        print(f"Error found with ScanImageTiffReader: '{e}'")
        return True
    return False


def main(mouse_name: str, date: str, cache_metadata: bool) -> None:
    """Looks for broken tiff files in a given session. It then caches all tiff metadata and saves fixed tiff files."""
    tiffs_dir = TIFF_UMBRELLA / date / mouse_name
    tiff_files = sorted(tiffs_dir.glob("*.tif")) + sorted(tiffs_dir.glob("*.tiff"))
    tiff_files_checked = list()
    broken_tiffs = list()
    if cache_metadata:
        stack_lengths = list()
        epochs = list()
        all_tiff_timestamps = list()
    for _, f in enumerate(tiff_files):
        broken = check_if_tiff_is_broken(f)
        if broken:
            broken_tiffs.append(f)
            tiff_files_checked.append((f, True))
            if cache_metadata:
                stack_length, epoch, timestamps = extract_metadata_broken_tiff(f)
                stack_lengths.append(stack_length)
                epochs.append(epoch)
                check_no_dropped_frames(timestamps)
                all_tiff_timestamps.extend(timestamps)
        else:
            tiff_files_checked.append((f, False))
            if cache_metadata:
                stack_length, epoch, timestamps = extract_metadata(
                    ScanImageTiffReader(str(f))
                )
                stack_lengths.append(stack_length)
                epochs.append(epoch)
                check_no_dropped_frames(timestamps)
                all_tiff_timestamps.extend(timestamps)
    print("All tiff files checked")
    if cache_metadata:
        for variable, name in zip(
            [stack_lengths, all_tiff_timestamps, epochs],
            ["stack_lengths", "all_tiff_timestamps", "epochs"],
        ):
            np.save(
                TEMP_CACHE_PATH / f"{mouse_name}_{date}_{name}.npy",
                variable,
            )
        print("Metadata extracted and cached")
    print(f"Broken tiffs: {broken_tiffs}")
    for path in broken_tiffs:
        save_tiff_until_broken(path)


if __name__ == "__main__":
    mouse_name = "JB018"
    dates = ["2024-11-27"]
    for date in dates:
        main(mouse_name, date, True)
