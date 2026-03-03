from dataclasses import dataclass
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from urllib.error import HTTPError
import pandas as pd
import pickle
from scipy import stats
import seaborn as sns

import statsmodels.formula.api as smf

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.constants import SPREADSHEET_ID
from viral.utils import get_genotype
from viral.gsheets_importer import gsheet2df


@dataclass
class SpatialInformation:
    mouse_id: str
    date: str
    rewarded: bool
    data: np.ndarray
    p_value: np.ndarray
    session_type: str


def parse_csv(path: Path, metadatas: dict[str, pd.DataFrame]) -> SpatialInformation:

    mouse_id = path.stem.split("_")[0]
    if mouse_id == "JB031":
        return None

    date = path.stem.split("_")[1]
    metadata = metadatas[mouse_id]
    row = metadata[metadata["Date"] == date]
    session_type = row["Type"].values[0]

    p_value_file = path.parent / f"{mouse_id}_{date}_p_values.csv"
    assert p_value_file.exists(), f"Missing p-value file for {mouse_id} on {date}"

    return SpatialInformation(
        mouse_id=mouse_id,
        date=date,
        session_type=session_type,
        rewarded=path.parent.stem == "rewarded",
        data=pd.read_csv(path).to_numpy().squeeze(),
        p_value=pd.read_csv(p_value_file).to_numpy().squeeze(),
    )


def preprocess_data() -> None:
    umbrella = Path("/Volumes/MarcBusche/Dan")
    si_files = list(umbrella.rglob("*_si_values.csv"))
    if not si_files:
        raise FileNotFoundError(
            "No spatial information files found in the specified directory."
        )

    max_mouse = 36
    min_mouse = 11

    metadatas: dict[str, pd.DataFrame] = {}
    for mouse_idx in range(min_mouse, max_mouse + 1):
        mouse = f"JB0{mouse_idx}"
        try:
            metadatas[mouse] = gsheet2df(SPREADSHEET_ID, mouse, 1)
        # Should catch the error properly but i can't find how to import it
        except:
            continue

    si_sessions = []
    for si_file in si_files:
        spatial_information = parse_csv(si_file, metadatas)
        if spatial_information is not None:
            si_sessions.append(spatial_information)

    with open("si_sessions.pkl", "wb") as f:
        pickle.dump(si_sessions, f)


def main() -> None:

    check = {}
    with open("si_sessions.pkl", "rb") as f:
        si_sessions = pickle.load(f)

    colors = sns.color_palette("Set2", 10)
    genotypes = [
        "WT",
        "NLGF",
        "Oligo-BACE1-KO",
    ]

    mice = set(session.mouse_id for session in si_sessions)
    mouse_dict = {mouse: [] for mouse in mice}
    for mouse in mice:
        for session in si_sessions:
            if session.mouse_id == mouse:
                mouse_dict[mouse].append(session)

    data = {
        "genotype": [],
        "stage": [],
        "si": [],
        "ratio": [],
        "mouse_id": [],
        "p_value": [],
    }

    for mouse, sessions in mouse_dict.items():
        all_dates = set(session.date for session in sessions)
        for date in all_dates:
            rewarded_session = next(
                session
                for session in sessions
                if session.date == date and session.rewarded
            )
            unrewarded_session = next(
                session
                for session in sessions
                if session.date == date and not session.rewarded
            )

            assert (
                rewarded_session.date == unrewarded_session.date
                and rewarded_session.mouse_id == unrewarded_session.mouse_id
                and rewarded_session.session_type == unrewarded_session.session_type
            )
            stage = rewarded_session.session_type.lower().split(" ")[0]
            genotype = get_genotype(mouse)

            if stage != "learning":
                continue
            if genotype == "Oligo-BACE1-KO":
                continue

            # sig = np.logical_and(
            #     rewarded.p_value < 0.05,
            #     unrewarded.p_value < 0.05,
            # )
            # rewarded = rewarded.data[sig]
            # unrewarded = unrewarded.data[sig]

            rewarded = rewarded_session.data
            unrewarded = unrewarded_session.data
            p_value = (rewarded_session.p_value + unrewarded_session.p_value) / 2

            division = rewarded / unrewarded
            division[np.isinf(division)] = 1
            division[np.isnan(division)] = 1
            data["genotype"].extend([genotype] * len(division))
            data["stage"].extend([stage] * len(division))
            data["mouse_id"].extend([mouse] * len(division))
            data["ratio"].extend(division.tolist())
            data["si"].extend((rewarded + unrewarded) / 2)
            data["p_value"].extend(p_value.tolist())

    data = pd.DataFrame(data)

    for dependent in ["si", "ratio"]:
        # model = smf.mixedlm(f"{dependent} ~ genotype", data, groups=data["mouse_id"])
        # result = model.fit()
        # p_value = result.pvalues["genotype[T.WT]"]
        ks_stat, p_value = stats.ks_2samp(
            data[data["genotype"] == "WT"][dependent],
            data[data["genotype"] == "NLGF"][dependent],
        )

        plt.figure()

        # sns.histplot(data=data, x="stage", y=dependent, hue="genotype")
        sns.histplot(
            data=data,
            x=dependent,
            hue="genotype",
            kde=True,
            stat="density",
            # log_scale=,
        )
        plt.title(f"{dependent} p-value (KS-test): {p_value:.3f}")

        plt.ylabel(
            "Total spatial information (bits)"
            if dependent == "si"
            else "Rewarded/Unrewarded SI Ratio"
        )
        if dependent == "ratio":
            plt.xlim(0, 5)

        plt.xlabel("Stage")

    1 / 0


if __name__ == "__main__":
    # preprocess_data()
    main()
