import sys
import numpy as np

from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

from viral.imaging_utils import compute_dff, subtract_neuropil

"""
prior, run 'pip install -e /path/to/Cascade', e.g. 'pip install -e /home/josef-bitzenhofer/code/Cascade/' with everything installed there
"""

from cascade2p import cascade

server_path = Path("/home/josef-bitzenhofer/mnt/MarcBusche/Josef/2P")


def get_dff(s2p_path: Path) -> np.ndarray:
    # TODO: eventually, find a way of harmonizing the retrieval of imaging data
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)

    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]

    return compute_dff(subtract_neuropil(f_raw, f_neu))


if __name__ == "__main__":
    mouse = "JB031"
    date = "2025-03-28"

    s2p_path = Path(server_path / date / mouse / "suite2p" / "plane0")
    print(f"Suite 2p path is {s2p_path}")

    dff = get_dff(s2p_path)

    model_name = "GC8_EXC_30Hz_smoothing50ms_high_noise"
    cascade.download_model(model_name, verbose=1)
    spike_prob = cascade.predict(model_name, dff, verbosity=1)

    np.save(s2p_path / "cascade_spk_prob.py", spike_prob)
