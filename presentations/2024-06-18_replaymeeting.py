from pathlib import Path
import sys


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

from matplotlib import pyplot as plt

from viral.constants import HERE
from viral.multiple_sessions import (
    get_chance_level,
    load_cache,
    plot_rolling_performance,
)


def plot_rolling_performance_all_mice() -> None:

    mice = [load_cache(mouse_name) for mouse_name in ["J004", "J005", "J007"]]

    # TODO: The range is about -0.5 to 0.5
    # chance = get_chance_level(mice)

    plt.figure(figsize=(4 * 3, 4))
    for idx, mouse in enumerate(mice):
        plt.subplot(1, 3, idx + 1)
        plt.title(f"wild type {idx + 1}")
        plot_rolling_performance(mouse.sessions, 50)
        plt.ylim(-0.8, 4.1)
        # plt.xlim(0, 600)
        plt.xlabel("trial number")
        if idx == 0:
            plt.ylabel("learning metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "rolling_performance_WT.png")

    plt.figure(figsize=(4 * 2, 4))
    for idx, mouse_name in enumerate(["J015", "J016"]):
        mouse = load_cache(mouse_name)
        plt.subplot(1, 2, idx + 1)
        plt.title(f"NLGF {idx + 1}")
        plot_rolling_performance(mouse.sessions, 50, idx == 0)
        plt.ylim(-0.8, 3)
        # plt.xlim(0, 600)
        plt.xlabel("trial number")
        if idx == 0:
            plt.ylabel("learning metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "rolling_performance_NLGF.png")


if __name__ == "__main__":
    plot_rolling_performance_all_mice()
