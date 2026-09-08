"""Decoding accuracy against accumulated task experience.

Rebuilds the decoder_over_sessions plot with three changes.

CUMULATIVE TRIALS come from behaviour_summaries, so sessions that were never imaged
still count toward experience - which is the point of the axis. Note what that source
does and does not contain: cache_mouse keeps only "learning day" type sessions and
skips habituation, water-in-dish and UNSUPERVISED sessions, so pre-training corridor
exposure is not counted. The original also counted only sessions whose type starts with
"learning day", which excludes reversal and recall; here all task stages count, since
experience accrues in those too.

MATCHED ACCURACY on the y axis. Unmatched decoding error tracks trials-in-session
(rho -0.33) and cell count (rho -0.42), and trials-in-session is itself a component of
cumulative trials - so an unmatched y against this x can produce a slope out of data
quantity alone. decoding_matched.csv fixes 24 trials and 100 cells per session.

BOTH AXES. NLGF run ~37% more trials per session, so cumulative trials advances faster
in real time for them: at any calendar date they sit further right. That makes the axis
choice decide the question.

    vs cumulative trials   "does experience refine the map?" - correct axis, and NLGF
                           genuinely have more experience
    vs days elapsed        "does pathology progress?" - correct axis, since cumulative
                           trials is now entangled with hyperlocomotion

Because NLGF pathology is age dependent, the two axes diverge systematically BY
GENOTYPE, so agreement between them is what makes a slope trustworthy.

Inference reuses dec.py unchanged: a mixed model with mouse random intercept and slope,
and the genotype label permuted across mice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.load import ledger
from viral.nlgf.paths import RESULTS


def experience_table() -> pd.DataFrame:
    beh = pd.read_csv(RESULTS / "behaviour_full.csv")
    dec = pd.read_csv(RESULTS / "decoding_matched.csv")
    dec = dec[dec.genotype.isin(["WT", "NLGF"])].copy()

    # Cumulative trials over every task session, imaged or not, in the order run
    beh = beh.sort_values(["mouse", "session_overall"]).copy()
    beh["cum_trials"] = beh.groupby("mouse")["n_trials"].cumsum()
    beh["key"] = beh.mouse + "|" + beh.session_name.str.lower().str.strip()
    # One duplicated (mouse, session_name); keep the first occurrence in run order
    beh = beh.drop_duplicates(subset="key", keep="first")

    dec["key"] = dec.mouse + "|" + dec.session_type.str.lower().str.strip()
    d = dec.merge(beh[["key", "cum_trials", "n_trials", "stage"]], on="key", how="left")
    d = d.rename(columns={"n_trials": "trials_in_session"})

    # Calendar axis: days since that mouse's first session of any kind
    led = ledger()
    first = (pd.to_datetime(led.date).groupby(led.mouse).min().rename("first_date"))
    d["date_dt"] = pd.to_datetime(d.date)
    d = d.join(first, on="mouse")
    d["days_elapsed"] = (d.date_dt - d.first_date).dt.days

    # Columns dec.py expects
    d["acc"] = d.decoding_error_matched
    d["n_cells"] = d.n_cells_total
    d = d.dropna(subset=["cum_trials", "acc", "trials_in_session", "n_cells",
                         "days_elapsed"])
    d.to_csv(RESULTS / "experience.csv", index=False)
    return d


def fit(d: pd.DataFrame, axis: str = "cum_trials", n_perm: Optional[int] = 400,
        random_slope: bool = True):
    """dec.py's model and permutation, with the experience axis swapped in.

    random_slope=False replaces dec.fit's random intercept AND slope by mouse with a
    random intercept only. The slope model asks 15 animals to support an intercept
    variance, a slope variance and their covariance, and it does not manage it: 483 of
    ~800 fits during the permutation reported "Maximum Likelihood optimization failed
    to converge". A non-converged fit still returns coefficients, so the permutation
    null is then built partly from garbage and the p-value means little.

    The rebinding is scoped to the call and dec.py is left untouched.
    """
    import contextlib

    import statsmodels.formula.api as smf

    import viral.dec as dec

    frame = d.copy()
    frame["cum_trials"] = frame[axis]
    prepared = dec.prepare(frame, logit_y=False)

    @contextlib.contextmanager
    def intercept_only():
        original = dec.fit
        dec.fit = lambda x: smf.mixedlm(dec.FORMULA, x, groups=x["mouse"]).fit(
            method="lbfgs")
        try:
            yield
        finally:
            dec.fit = original

    if random_slope:
        return dec.permutation_test(prepared, n_perm=n_perm)
    with intercept_only():
        return dec.permutation_test(prepared, n_perm=n_perm)


if __name__ == "__main__":
    d = experience_table()
    print(f"{len(d)} sessions, {d.mouse.nunique()} mice")
    print(d.groupby("genotype").agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        cum_trials_max=("cum_trials", "max"), days_max=("days_elapsed", "max"),
        error=("acc", "median")).to_string())

    print("\n=== trials accumulated per day elapsed (the axis divergence) ===")
    rate = d.assign(rate=d.cum_trials / d.days_elapsed.clip(lower=1))
    print(rate.groupby("genotype")["rate"].median().round(2).to_string())

    for axis in ("cum_trials", "days_elapsed"):
        for slope in (True, False):
            tag = "random intercept + slope" if slope else "random intercept only"
            print(f"\n=== decoding error vs {axis}  ({tag}) ===", flush=True)
            print(fit(d, axis, random_slope=slope).to_string(index=False), flush=True)
