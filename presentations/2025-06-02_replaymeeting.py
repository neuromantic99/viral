import json
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

from collections import defaultdict

from viral.utils import get_genotype
from viral.learning_stages import get_mouse_sessions


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
    "JB031",  # WT
]


def grosmark_plot() -> None:
    """
    Placeholder function for Grosmark plot.
    """
    print("Grosmark plot function is not implemented yet.")


if __name__ == "__main__":

    genotype_dict = defaultdict(list)

    for mouse_name in MICE_FOR_PRESENTATION:
        if mouse_name in ["JB014", "JB015", "JB016", "JB018", "JB019"]:
            continue
        genotype = get_genotype(mouse_name)
        sessions = get_mouse_sessions(mouse_name)
        # try:
        #     sessions = get_mouse_sessions(mouse_name)
        # except Exception as e:
        #     print(f"Error retrieving sessions for {mouse_name}: {e}")
        #     continue

        genotype_dict[genotype].append(mouse_name)

    with open("replay_meeting_data.json", "w") as f:
        json.dump(genotype_dict, f)

    assert len(genotype_dict) == 3, "Expected 3 genotypes"
    assert all(
        len(v) == 3 for v in genotype_dict.values()
    ), "Expected 3 mice per genotype"
