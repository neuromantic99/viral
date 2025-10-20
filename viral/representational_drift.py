from pathlib import Path
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, TIFF_UMBRELLA
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.utils import get_wheel_circumference_from_rig
from viral.sessions_keep import SESSIONS_KEEP


from sklearn.linear_model import LogisticRegression


def do_classify(session: Cached2pSession, spks: np.ndarray) -> None:

    bin_size = 2
    start = 0
    max_position = 180

    X_list = []
    y = []

    for trial in session.trials:
        if not trial_is_imaged(trial):
            continue

        data = activity_trial_position(
            trial=trial,
            flu=spks,
            wheel_circumference=get_wheel_circumference_from_rig("2P"),
            bin_size=bin_size,
            start=start,
            max_position=max_position,
            verbose=False,
            do_shuffle=False,
            threshold_speed=False,
        )
        X_list.append(data)
        if trial.texture_rewarded:
            # if random.random() > 0.5:
            y.append(1)
        else:
            y.append(0)

    X = np.array(X_list)

    scores_position = []

    coefs = []
    for bin in range(X.shape[2]):

        scores = []
        for fold in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, :, bin], y, test_size=0.2
            )

            clf = LogisticRegression(penalty="l1", solver="liblinear").fit(
                X_train, y_train
            )

            scores.append(clf.score(X_test, y_test))
            coefs.append(clf.coef_)

        scores_position.append(scores)

    scores_position = np.array(scores_position)
    coefs = np.array(coefs)

    x_axis = np.arange(start, max_position, bin_size)
    plt.axhline(0.5)

    mean = np.mean(scores_position, 1)
    std = np.std(scores_position, 1)
    plt.plot(x_axis, mean)
    plt.fill_between(x_axis, mean - std, mean + std, alpha=0.5)

    plt.xlabel("Position (cm)")
    plt.ylabel("Classifier accuracy")
    plt.ylim(0, 1)
    1 / 0


if __name__ == "__main__":
    mouse_name = "JB030"

    date = SESSIONS_KEEP[mouse_name]["learning"]
    path = CACHE_PATH / f"{mouse_name}_{date}.json"

    spks_path = TIFF_UMBRELLA / date / mouse_name / "suite2p" / "plane0"
    spks = np.load(spks_path / "oasis_spikes.npy")
    session = Cached2pSession.model_validate_json(path.read_text())
    do_classify(session, spks)
