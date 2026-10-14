"""Local-only paths. The server (/Volumes/MarcBusche) is deliberately never touched.

CACHE_PATH and friends in viral.constants point at the server, so any function that
builds its own cache path there cannot be called from this package - see digest.py
and place_cells.py for local reimplementations of the loading, which reuse the
computation from viral/* unchanged.
"""

from pathlib import Path

HD = Path("/Volumes/hard_drive/viral")

CACHED_2P = HD / "cached_2p"
DFF = HD / "dff"
PLACE_THRESHOLD = HD / "place_threshold"

# Everything this package writes lives here
DERIVED = HD / "nlgf_derived"
DIGESTS = DERIVED / "digests"
RESULTS = DERIVED / "results"

# Plots go in the repo, next to the existing ones
PLOTS = Path(__file__).parent.parent.parent / "plots" / "nlgf"

for _p in (DERIVED, DIGESTS, RESULTS, PLOTS):
    _p.mkdir(parents=True, exist_ok=True)


def spks_path(mouse: str, date: str) -> Path:
    return DFF / f"{mouse}_{date}_spks.npy"


def dff_path(mouse: str, date: str) -> Path:
    return DFF / f"{mouse}_{date}_dff.npy"


def denoised_path(mouse: str, date: str) -> Path:
    return DFF / f"{mouse}_{date}_denoised.npy"


def digest_path(mouse: str, date: str) -> Path:
    return DIGESTS / f"{mouse}_{date}.npz"


def session_json_paths() -> list[Path]:
    """Cached session JSONs, minus the AppleDouble ._ stubs that copying to exFAT
    leaves behind. Those are not JSON and raise UnicodeDecodeError if parsed."""
    return sorted(
        p for p in CACHED_2P.glob("*.json") if not p.name.startswith("._")
    )
