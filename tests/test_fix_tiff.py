from pathlib import Path
import numpy as np

from viral.cache_2p_sessions import extract_metadata
from viral.constants import SERVER_PATH
from viral.fix_tiff import extract_metadata_broken_tiff

from ScanImageTiffReader import ScanImageTiffReader


def test_extract_metadata_broken_tiff() -> None:
    """Don't have this as part of the test suite as it loads a file from the server"""

    tiff_path = Path(
        SERVER_PATH.parent
        / "James/Regular2p/2024-02-01/2024-02-01_J001_50mW_1x_BothHem_00001.tif"
    )

    metadata = extract_metadata(ScanImageTiffReader(str(tiff_path)))
    diffed = np.diff(metadata[2])
    assert np.round(np.max(diffed), 3) == np.round(np.min(diffed), 3) == 0.033

    broken_metadata = extract_metadata_broken_tiff(tiff_path)
    assert broken_metadata[0] + 1 == metadata[0]
    assert broken_metadata[1] == metadata[1]
    assert np.array_equal(broken_metadata[2], metadata[2][:-1])
