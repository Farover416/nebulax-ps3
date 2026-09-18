"""PS3 predictors. One call per subsystem, uniform shape.

    from ps3 import predict
    predict.shm("data/test01.csv")    -> {"prediction": 0.41, "detail": {...}}
    predict.door("data/Test.csv")     -> {"segments": DataFrame, "detail": {...}}
    predict.acv("data/case.xlsx")     -> {"ranked_cars": "03|01|...", "detail": {...}}
    predict.rail("data/Test1.csv")    -> {"prediction": "Side I", "detail": {...}}

Or dispatch by name, which is what the FastAPI route should do:

    predict.run("shm", path)

Every result carries a `detail` dict meant to be rendered directly in the UI.
Nothing here trains; fit_shm.py and fit_rail.py produce the model files.

Command line (from the repo root), for a single file or a whole folder:

    python -m ps3.predict --subsystem shm --input SHM/Test --output shm_predictions.csv
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


def main(argv=None) -> None:
    """Write a submission-format CSV for one file or every matching file in a folder."""
    import argparse
    import glob
    import os

    import pandas as pd

    ap = argparse.ArgumentParser(description="Run a PS3 predictor and write its predictions CSV.")
    ap.add_argument("--subsystem", required=True, choices=sorted(SUBSYSTEMS))
    ap.add_argument("--input", required=True, help="a data file, or a folder of them")
    ap.add_argument("--output", required=True, help="CSV to write, e.g. shm_predictions.csv")
    a = ap.parse_args(argv)

    accepts = SUBSYSTEMS[a.subsystem]["accepts"]
    files = [a.input] if os.path.isfile(a.input) else sorted(
        f for f in glob.glob(os.path.join(a.input, "*")) if os.path.splitext(f)[1].lower() in accepts)
    if not files:
        raise SystemExit(f"no {'/'.join(accepts)} files found in {a.input}")

    if a.subsystem == "door":
        out = pd.concat([run("door", f)["segments"] for f in files], ignore_index=True)
    elif a.subsystem == "acv":
        out = pd.DataFrame([{"file_id": os.path.basename(f), "ranked_cars": run("acv", f)["ranked_cars"]}
                            for f in files])
    else:
        out = pd.DataFrame([{"file_id": os.path.basename(f), "prediction": run(a.subsystem, f)["prediction"]}
                            for f in files])
    out.to_csv(a.output, index=False)
    print(f"{a.subsystem}: {len(files)} file(s) -> {a.output}")


if __name__ == "__main__":
    main()
