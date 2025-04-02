import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.utils import get_wheel_circumference_from_rig


from sklearn.linear_model import LogisticRegression


def do_classify(session: Cached2pSession, spks: np.ndarray) -> None:

    bin_size = 10
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
            smoothing_sigma=None,
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
    plt.show()
