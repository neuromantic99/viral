import sys

from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

import json
import pickle
from scipy.stats import zscore

from typing import Any, Dict

from viral.grosmark_analysis import binarise_spikes, grosmark_place_field
from viral.imaging_utils import load_only_spks
from viral.models import GrosmarkConfig, Mouse2pSessions

import seaborn as sns

from collections import defaultdict

from viral.utils import get_genotype, session_is_unsupervised, shaded_line_plot
from viral.learning_stages import get_completed_mouse_sessions, get_mouse_sessions


from viral.learning_stages import SESSIONS_KEEP

MICE_FOR_PRESENTATION = [
    "JB014",  # B1
    "JB015",  # B1
    "JB016",  # NLGF
    "JB018",  # B1
    "JB019",  # NLGF
    "JB021",  # NLGF
    "JB026",  # WT
    "JB030",  # WT
    "JB033",  # WT
]


def already_done_grosmark(session: Any, rewarded: bool) -> bool:
    results_path = Path(
        "/Users/jamesrowland/Code/viral/results/pairwise-reactivations-ITI"
    )

    for file in results_path.glob(f"*.npy"):
        # This won't work if you change the config
        if file.stem.startswith(
            f"{session.mouse_name}_{session.date}_rewarded_{rewarded}"
        ):
            print(f"File {file} already exists, skipping Grosmark analysis")
            return True

    return False


def grosmark_plot(genotype_dict: Dict) -> None:

    config = GrosmarkConfig(
        bin_size=2,
        start=30,
        end=160,
    )
    for genotype, mouse_2p_sessions in genotype_dict.items():

        for mouse in mouse_2p_sessions:
            if mouse.mouse_name == "JB014":
                continue
            for session in [mouse.unsupervised, mouse.learning, mouse.learned]:
                is_unsupervised = session_is_unsupervised(session)
                rewarded = None if is_unsupervised else False

                if already_done_grosmark(session, rewarded):
                    print(
                        f"Skipping {session.mouse_name} on {session.date} as Grosmark analysis already done."
                    )
                    continue

                print(
                    f"Processing {session.mouse_name} on {session.date} with genotype {genotype}"
                )

                spks = load_only_spks(session.mouse_name, session.date)
                spks = binarise_spikes(spks)

                try:
                    grosmark_place_field(
                        session,
                        spks,
                        rewarded=None if is_unsupervised else False,
                        config=config,
                    )
                except Exception as e:
                    print(
                        f"Error processing {session.mouse_name} on {session.date}: {e}"
                    )
                    continue


def grosmark_summary_plot() -> None:
    npy_path = Path("/Users/jamesrowland/Code/viral/results/pairwise-reactivations-ITI")
    to_plot = {}
    for file in npy_path.glob("*.npy"):

        mouse = file.stem.split("_")[0]
        date = file.stem.split("_")[1]
        genotype = get_genotype(mouse)

        try:
            lookup = {y: x for x, y in SESSIONS_KEEP[mouse].items()}
        except KeyError:
            print(f"Mouse {mouse} not found in SESSIONS_KEEP, skipping.")
            continue
        session_type = lookup[date]

        if session_type == "unsupervised":
            continue
        data = np.load(file)
        x_axis = data[0, :]

        if genotype not in to_plot:
            to_plot[genotype] = [data[1, :]]
        else:
            to_plot[genotype].append(data[1, :])

    colors = ["red", "blue", "green"]
    color_idx = 0
    for genotype, data in to_plot.items():
        plt.figure()
        data_plot = np.array(data)
        shaded_line_plot(
            arr=zscore(np.array(data), axis=1),
            x_axis=x_axis,
            color=colors[color_idx],
            label=genotype,
        )
        plt.xlim(0, 70)
        plt.ylim(-1.5, 2.5)
        color_idx += 1

        plt.legend()
        plt.xlabel("Distance between peaks (cm)")
        plt.ylabel("Pearson correlation (z-scored)")
        plt.tight_layout()
        plt.savefig(f"grosmark_summary_{genotype}.png")

    plt.show()


if __name__ == "__main__":

    grosmark_summary_plot()
    # genotype_dict = pickle.load(open("replay_meeting_data.pkl", "rb"))
    # grosmark_plot(genotype_dict)

    # genotype_dict = defaultdict(list)

    # for mouse_name in MICE_FOR_PRESENTATION:

    #     genotype = get_genotype(mouse_name)
    #     sessions = get_completed_mouse_sessions(mouse_name)

    # try:
    #     sessions = get_mouse_sessions(mouse_name)
    # except Exception as e:
    #     print(f"Error retrieving sessions for {mouse_name}: {e}")
    #     continue

    # genotype_dict[genotype].append(sessions)

    # with open("replay_meeting_data.pkl", "wb") as f:
    #     pickle.dump(genotype_dict, f)

    # assert len(genotype_dict) == 3, "Expected 3 genotypes"
    # assert all(
    #     len(v) == 3 for v in genotype_dict.values()
    # ), "Expected 3 mice per genotype"
