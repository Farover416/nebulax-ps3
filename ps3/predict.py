"""PS3 predictors. One call per subsystem, uniform shape.

    from ps3 import predict
    predict.shm("data/test01.csv")    -> {"prediction": 0.41, "detail": {...}}
    predict.door("data/Test.csv")     -> {"segments": DataFrame, "detail": {...}}
    predict.acv("data/case.xlsx")     -> {"ranked_cars": "03|01|...", "detail": {...}}
    predict.rail("data/Test1.csv")    -> {"prediction": "Side I", "detail": {...}}

Or dispatch by name, which is what the FastAPI route should do:

    predict.run("shm", path)

Every result carries a `detail` dict meant to be rendered directly in the UI.
Nothing here trains; fit_shm.py and fit_rail.py produce the two artifacts.
"""
from . import acv as _acv
from . import door as _door
from . import rail as _rail
from . import shm as _shm

SUBSYSTEMS = {
    "shm": {
        "fn": _shm.predict,
        "label": "Structural Health Monitoring",
        "accepts": [".csv"],
        "returns": "cumulative fatigue damage (0-1)",
        "metric": "max(0, 1 - MAPE)",
    },
    "door": {
        "fn": _door.predict,
        "label": "Train Door",
        "accepts": [".csv"],
        "returns": "one row per detected cycle: start_time, end_time, prediction",
        "metric": "IoU-weighted F1",
    },
    "acv": {
        "fn": _acv.predict,
        "label": "Air Conditioning & Ventilation",
        "accepts": [".xlsx", ".xls"],
        "returns": "all cars ranked most to least likely, pipe-separated",
        "metric": "rank decay (n - (r-1)) / n",
    },
    "rail": {
        "fn": _rail.predict,
        "label": "Rail Corrugation",
        "accepts": [".csv"],
        "returns": "Normal | Side I | Side II",
        "metric": "macro F1",
    },
}


def run(subsystem: str, path: str) -> dict:
    key = subsystem.strip().lower()
    if key not in SUBSYSTEMS:
        raise ValueError(f"unknown subsystem {subsystem!r}; "
                         f"expected one of {sorted(SUBSYSTEMS)}")
    return SUBSYSTEMS[key]["fn"](path)


shm, door, acv, rail = _shm.predict, _door.predict, _acv.predict, _rail.predict

__all__ = ["run", "shm", "door", "acv", "rail", "SUBSYSTEMS"]
