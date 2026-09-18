"""ACV: identify which car has the refrigerant leak, as a ranked list.

Physics first: a refrigerant leak means lost cooling capacity, which means
that car's cabin runs warmer than its siblings under the same ambient and
the same duty cycle. Taking the deviation from the cross-car median at each
timestamp cancels ambient temperature, time of day and route entirely.

Score per car = mean over time of ( its indoor temp - median indoor temp
across all cars at that timestamp ).

On the six training cases this ranks the true faulty car 1st five times and
2nd once -> rank-decay score 0.979. A random permutation scores 0.5625, so
the floor is high and the method clears it comfortably.

Honest caveat for the write-up: two of the six margins are thin (0.018 C and
0.085 C), so expect 0.88-1.00 on a single held-out file rather than 0.979.
Six candidate scores were compared against six cases, which is selection on
a tiny sample -- the reason to trust this one is that it was the physics
prediction going in, not the winner of a search.
"""
import re

import numpy as np
import pandas as pd

CAR_COL = re.compile(r"^Car (\d{2}) - (.+)$")

# case_04 uses a different vocabulary for the same quantities
ALIAS = {
    "Passenger Cabin Temperature Detected Value": "Indoor Average Temperature",
    "Outside Temperature Sensor Reading": "Outdoor Average Temperature",
    "Target Temperature Value": "ACV Control Temperature (Cooling)",
}


def _wide(df: pd.DataFrame, param: str) -> pd.DataFrame | None:
    cols = {}
    for c in df.columns:
        m = CAR_COL.match(c)
        if m and ALIAS.get(m.group(2), m.group(2)) == param:
            cols[m.group(1)] = df[c]
    if not cols:
        return None
    return pd.DataFrame(cols)[sorted(cols)]


def predict(path: str) -> dict:
    df = pd.read_excel(path) if isinstance(path, str) else path

    ind = _wide(df, "Indoor Average Temperature")
    if ind is None:
        raise ValueError("no per-car indoor temperature column found")
    I = ind.apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    dev = I.sub(I.median(axis=1), axis=0).mean()

    # tiebreaker: crew switching a car to Manual Control is a corroborating
    # signal (0.925 across the six training cases) but never overrides the
    # temperature evidence
    setm = _wide(df, "ACV Setting Mode")
    manual = (setm.astype(str) == "Manual Control").mean() if setm is not None \
        else pd.Series(0.0, index=dev.index)

    order = pd.DataFrame({"dev": dev, "manual": manual.reindex(dev.index).fillna(0)})
    order = order.sort_values(["dev", "manual"], ascending=False)

    ranked = list(order.index)
    margin = float(order["dev"].iloc[0] - order["dev"].iloc[1]) if len(order) > 1 else np.nan

    return {
        "ranked_cars": "|".join(ranked),
        "prediction": ranked[0],
        "detail": {
            "scores_degC": {c: round(float(v), 4) for c, v in order["dev"].items()},
            "margin_over_runner_up_degC": round(margin, 4),
            "confidence": "high" if margin >= 0.15 else
                          "medium" if margin >= 0.05 else "low",
            "manual_control_fraction": {c: round(float(v), 3)
                                        for c, v in order["manual"].items()},
            "note": f"Car {ranked[0]} runs {margin:.3f} C warmer than the next "
                    "car relative to the train median, consistent with reduced "
                    "cooling capacity from a refrigerant leak.",
        },
    }
