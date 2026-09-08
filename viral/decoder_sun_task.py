from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User parameters
# ============================================================
DATA_ROOT = Path(r"Y:\Dan")
mouse = "JB026"
DATES = ["2024-12-14", "2024-12-15"]  # day1 train, day2 test
ANALYSIS_SUBDIR = Path("goodwinAnalysis") / "rewarded"

CELLREG_BASE = Path(r"C:\Users\dg453\OneDrive - University of Exeter\Desktop\UCL\Dan")
CELLREG_MATCH = "*cellRegistered*.mat"

# Track bins
track_min_cm = 10
track_max_cm = 170
pos_bin_cm   = 2.0

# Decoder bins
fps = 30.0
dt = 0.1
frames_per_bin = int(np.round(fps * dt))
assert frames_per_bin >= 1

# Event definition (ONLY used if you don't already have an event raster)
EVENT_Z = 2.0  # z-score threshold per neuron within session

# Subsampling
N_SAMPLES = 100
N_NEURONS = 150
RNG_SEED  = 0

# Optional: "odd laps" training (for day1 internal validation style)
TRAIN_ON_ODD_LAPS = False  # if True: day1 uses odd laps only to train
# ============================================================


# -------------------------
# Paths / loading
# -------------------------
def session_dir(date: str) -> Path:
    return DATA_ROOT / date / mouse / ANALYSIS_SUBDIR

def beh_path(date: str) -> Path:
    return session_dir(date) / f"{mouse}_{date}_trimmed_behaviour.csv"

def fc_path_npy(date: str) -> Path:
    return session_dir(date) / f"{mouse}_{date}_Fc3_cleaned.npy"

def fc_path_csv(date: str) -> Path:
    return session_dir(date) / f"{mouse}_{date}_Fc3_cleaned.csv"


def load_beh(date: str) -> pd.DataFrame:
    p = beh_path(date)
    if not p.exists():
        raise FileNotFoundError(f"Missing behaviour CSV: {p}")
    df = pd.read_csv(p)
    if "position_cm" not in df.columns:
        raise ValueError(f"{mouse} {date}: behaviour missing 'position_cm'")
    df["position_cm"] = pd.to_numeric(df["position_cm"], errors="coerce")
    df = df.dropna(subset=["position_cm"]).reset_index(drop=True)
    return df

def load_fc(date: str) -> np.ndarray:
    p_npy = fc_path_npy(date)
    p_csv = fc_path_csv(date)

    if p_npy.exists():
        X = np.load(p_npy)  # (cells, frames)
    elif p_csv.exists():
        X = pd.read_csv(p_csv, header=None).to_numpy(float)
    else:
        raise FileNotFoundError(
            f"Could not find Fc3_cleaned for {mouse} {date}.\n"
            f"Tried:\n  {p_npy}\n  {p_csv}\n"
            f"Edit fc_path_* to match your filenames."
        )
    if X.ndim != 2:
        raise ValueError(f"Fc3_cleaned should be 2D (cells x frames). Got {X.shape}")
    return X


# -------------------------
# CellReg helpers
# -------------------------
def find_cellregistered_mat(cellreg_dir: Path, pattern: str) -> Path:
    mats = sorted(cellreg_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mats:
        raise FileNotFoundError(f"No .mat files matching {pattern} in {cellreg_dir}")
    return mats[0]

def load_cell_to_index_map(cellreg_mat_path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
        md = loadmat(cellreg_mat_path, struct_as_record=False, squeeze_me=True)
        crs = md["cell_registered_struct"]
        c2i = np.array(getattr(crs, "cell_to_index_map")).astype(int)
        if c2i.ndim != 2:
            raise ValueError(f"cell_to_index_map shape unexpected: {c2i.shape}")
        return c2i
    except NotImplementedError:
        import h5py
        with h5py.File(cellreg_mat_path, "r") as f:
            arr = np.array(f["cell_registered_struct"]["cell_to_index_map"]).astype(float)
            c2i = arr.astype(int)
            if c2i.shape[0] < c2i.shape[1] and c2i.shape[0] <= 10:
                c2i = c2i.T
            return c2i


# -------------------------
# Lap segmentation (for odd/even option)
# -------------------------
def lap_boundaries_from_position_drop(pos_cm: np.ndarray, drop_cm: float = 50.0) -> np.ndarray:
    pos_cm = np.asarray(pos_cm, float)
    dpos = np.diff(pos_cm)
    resets = np.where(dpos < -drop_cm)[0] + 1
    starts = np.concatenate(([0], resets))
    ends = np.concatenate((resets, [pos_cm.size]))
    return np.stack([starts, ends], axis=1)

def lap_id_vector(pos_cm: np.ndarray, drop_cm: float = 50.0) -> np.ndarray:
    laps = lap_boundaries_from_position_drop(pos_cm, drop_cm=drop_cm)
    lid = np.full(pos_cm.size, -1, int)
    for k, (s, e) in enumerate(laps):
        lid[s:e] = k
    return lid


# -------------------------
# Core: build event raster + compute g_{i,j} from training
# -------------------------
def to_event_raster_from_fc(X: np.ndarray, z_thresh: float = 2.0) -> np.ndarray:
    """
    Turn Fc3_cleaned (cells x frames) into a binary 'significant transient' raster.

    NOTE: If you already have a real event raster (deconvolved spikes / significant events),
    use that instead and skip this function.
    """
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    E = (Z > z_thresh).astype(np.int8)
    E[~np.isfinite(Z)] = 0
    return E

def make_position_bins(track_min, track_max, bin_cm):
    edges = np.arange(track_min, track_max + bin_cm, bin_cm)
    if edges[-1] < track_max:
        edges = np.append(edges, track_max)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers

def compute_g_rates(E_train: np.ndarray, pos_train: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    E_train: (M, T) binary events per frame
    pos_train: (T,) position cm
    edges: position bin edges
    Returns:
      g: (n_pos_bins, M)  rate of significant frames per second in each spatial bin
      pX: (n_pos_bins,)   occupancy prior (normalised)
    """
    M, T = E_train.shape
    pos = np.asarray(pos_train, float)

    # restrict to track range
    m = np.isfinite(pos) & (pos >= edges[0]) & (pos <= edges[-1])
    pos = pos[m]
    E = E_train[:, m]
    if E.shape[1] < 50:
        raise RuntimeError("Too few valid training frames in track range.")

    n_pos_bins = len(edges) - 1
    b = np.digitize(pos, edges) - 1
    b = np.clip(b, 0, n_pos_bins - 1)

    # occupancy in FRAMES
    occ_frames = np.bincount(b, minlength=n_pos_bins).astype(float)

    # count events per (posbin, neuron) in FRAMES
    # g wants rate per second: (events / time_in_bin_seconds)
    g = np.zeros((n_pos_bins, M), float)

    for i in range(n_pos_bins):
        mi = (b == i)
        if not np.any(mi):
            continue
        # events per neuron in this spatial bin (frames)
        ev = np.sum(E[:, mi], axis=1).astype(float)  # (M,)
        time_sec = mi.sum() / fps
        if time_sec > 0:
            g[i, :] = ev / time_sec

    # occupancy prior
    pX = occ_frames / np.sum(occ_frames) if np.sum(occ_frames) > 0 else np.ones(n_pos_bins) / n_pos_bins
    return g, pX


# -------------------------
# Core: decode (log domain, exact model)
# -------------------------
def decode_session(
    E_test: np.ndarray,
    pos_test: np.ndarray,
    g: np.ndarray,
    pX: np.ndarray,
    *,
    edges: np.ndarray,
    centers: np.ndarray,
    frames_per_bin: int,
    n_samples: int,
    n_neurons: int,
    rng: np.random.Generator,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      decoded_pos_avg: (n_timebins,) averaged decoded position across subsamples
      true_pos_bin:    (n_timebins,) true position (mean pos within timebin)
      t_sec:           (n_timebins,) time in seconds for each timebin (starts at 0)
    """
    M, T = E_test.shape
    n_pos_bins = g.shape[0]
    assert g.shape == (n_pos_bins, M)

    pos = np.asarray(pos_test, float)

    # Build time bins of size frames_per_bin
    n_timebins = T // frames_per_bin
    if n_timebins < 10:
        raise RuntimeError("Too few decoding time bins. Check fps/dt or data length.")

    # reshape into (M, n_timebins, frames_per_bin) and sum -> n_j per timebin
    E_trim = E_test[:, :n_timebins * frames_per_bin]
    n_counts = E_trim.reshape(M, n_timebins, frames_per_bin).sum(axis=2)  # (M, n_timebins)

    # "true position" for each timebin: mean position across frames
    pos_trim = pos[:n_timebins * frames_per_bin]
    true_pos = pos_trim.reshape(n_timebins, frames_per_bin).mean(axis=1)

    # Time axis in seconds for each bin
    t_sec = np.arange(n_timebins) * dt

    # mask out timebins whose true pos is out of range (optional)
    in_rng = np.isfinite(true_pos) & (true_pos >= edges[0]) & (true_pos <= edges[-1])

    # Precompute log(pX)
    log_pX = np.log(np.maximum(pX, 1e-12))

    # For each subsample, decode argmax_i log p(x_i|n)
    decoded_pos_samples = np.full((n_samples, n_timebins), np.nan, float)

    for s in range(n_samples):
        idx = rng.choice(M, size=n_neurons, replace=False)

        # Pull sub-matrices
        g_sub = g[:, idx]             # (n_pos_bins, n_neurons)
        n_sub = n_counts[idx, :]      # (n_neurons, n_timebins)

        # log likelihood:
        # log p(x_i|n) = log pX(x_i) + sum_j n_j log g_{i,j} - dt * sum_j g_{i,j}  (+ const)
        g_safe = np.maximum(g_sub, 1e-12)
        log_g = np.log(g_safe)

        # termA: sum_j n_j log g_{i,j}
        termA = log_g @ n_sub

        # termB: -dt * sum_j g_{i,j}
        termB = -dt * np.sum(g_sub, axis=1)[:, None]  # (n_pos_bins, 1)

        log_post = log_pX[:, None] + termA + termB

        decoded_bins = np.argmax(log_post, axis=0)  # (n_timebins,)
        decoded_pos_samples[s, :] = centers[decoded_bins]

    decoded_pos_avg = np.nanmean(decoded_pos_samples, axis=0)

    # apply in-range mask as NaN for clean error calc
    decoded_pos_avg[~in_rng] = np.nan
    true_pos[~in_rng] = np.nan

    return decoded_pos_avg, true_pos, t_sec


# ============================================================
# Run: load, match neurons, train day1, decode day2
# ============================================================
d1, d2 = DATES

beh1 = load_beh(d1)
beh2 = load_beh(d2)
pos1 = beh1["position_cm"].to_numpy(float)
pos2 = beh2["position_cm"].to_numpy(float)

X1_full = load_fc(d1)
X2_full = load_fc(d2)

# CellReg match
earliest_date = sorted(DATES)[0]
cellreg_dir = CELLREG_BASE / earliest_date / mouse
cellreg_mat = find_cellregistered_mat(cellreg_dir, CELLREG_MATCH)
print("Using CellReg file:", cellreg_mat)

c2i = load_cell_to_index_map(cellreg_mat)
if c2i.shape[1] < 2:
    raise RuntimeError(f"CellReg has {c2i.shape[1]} sessions but you requested 2")

col1 = c2i[:, 0]
col2 = c2i[:, 1]
valid = (col1 > 0) & (col2 > 0)
idx1 = col1[valid] - 1
idx2 = col2[valid] - 1

X1 = X1_full[idx1, :]
X2 = X2_full[idx2, :]

# sanity: lengths
if X1.shape[1] != pos1.shape[0]:
    raise ValueError(f"{mouse} {d1}: Fc frames {X1.shape[1]} != pos length {pos1.shape[0]}")
if X2.shape[1] != pos2.shape[0]:
    raise ValueError(f"{mouse} {d2}: Fc frames {X2.shape[1]} != pos length {pos2.shape[0]}")

M = X1.shape[0]
print("Matched neurons:", M)

if M < N_NEURONS:
    raise RuntimeError(f"Exclude: only {M} matched neurons (< {N_NEURONS}).")

# Convert to event rasters (swap this out if you have real significant-event rasters)
E1 = to_event_raster_from_fc(X1, z_thresh=EVENT_Z)  # (M, T1)
E2 = to_event_raster_from_fc(X2, z_thresh=EVENT_Z)  # (M, T2)

# Optional: train on odd laps only (day1)
if TRAIN_ON_ODD_LAPS:
    lid1 = lap_id_vector(pos1, drop_cm=50.0)
    odd_mask = (lid1 % 2) == 1
    E1_train = E1[:, odd_mask]
    pos1_train = pos1[odd_mask]
else:
    E1_train = E1
    pos1_train = pos1

edges, centers = make_position_bins(track_min_cm, track_max_cm, pos_bin_cm)

# Train: compute g_{i,j} and prior pX
g, pX = compute_g_rates(E1_train, pos1_train, edges)  # g: (n_pos_bins, M)

# Decode day2 with subsampling
rng = np.random.default_rng(RNG_SEED)
decoded_pos, true_pos, t_sec = decode_session(
    E2, pos2, g, pX,
    edges=edges, centers=centers,
    frames_per_bin=frames_per_bin,
    n_samples=N_SAMPLES,
    n_neurons=N_NEURONS,
    rng=rng,
    dt=dt,
)

# Error metric: mean absolute error across session
abs_err = np.abs(decoded_pos - true_pos)
mae = np.nanmean(abs_err)
print(f"Mean absolute error (cm): {mae:.2f}")

# Plot (x = actual time in seconds)
plt.figure(figsize=(10, 4))
plt.plot(t_sec, true_pos, label="True pos (day2, binned)", linewidth=1)
plt.plot(t_sec, decoded_pos, label=f"Decoded pos (avg of {N_SAMPLES}×{N_NEURONS})", linewidth=1, alpha=0.9)
plt.xlabel(f"Time (s))
plt.ylabel("Position (cm)")
plt.title(f"{mouse}: train {d1} → decode {d2} | MAE={mae:.2f} cm")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5.5, 4.5))
plt.hist(abs_err[np.isfinite(abs_err)], bins=40)
plt.xlabel("|Decoded − True| (cm)")
plt.ylabel("Time bins")
plt.title("Decoding error distribution")
plt.tight_layout()
plt.show()