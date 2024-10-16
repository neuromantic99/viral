from pathlib import Path
import sys


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from matplotlib import pyplot as plt
import numpy as np

from viral.utils import threshold_detect_edges


from viral.constants import SERVER_PATH


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


scope = "Regular2p"
date = "2024-09-18"
mouse = "J022"

s2p_path = SERVER_PATH / scope / date / mouse / "suite2p" / "plane0"
iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
spks = np.load(s2p_path / "spks.npy")[iscell, :]
f_raw = np.load(s2p_path / "F.npy")[iscell, :]
f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]


dff = compute_dff(subtract_neuropil(f_raw, f_neu))


# dff = dff - np.expand_dims(np.min(dff, 1), 1)


# above = []
# below = []

# for cell in range(dff.shape[0]):

#     above_cell, below_cell = threshold_detect_edges(dff[cell, :], 1)
#     above.append(above_cell)
#     below.append(below_cell)


# for i in range(50):
#     plt.plot(np.arange(dff.shape[1]) / 30, dff[i, :] + i * 1)


1 / 0
plt.show()
