"""Door: find each open/close cycle in the continuous stream, then classify
Normal vs Abnormal resistance.

Two findings this is built on:

1. Segmentation is free. Splitting on time gaps > 1 s reproduces all 110
   ground-truth segments in Train.csv with byte-identical start times, end
   times and row counts. Every IoU is 1.000, which collapses the official
   IoU-weighted F1 to plain binary accuracy.

2. The threshold must be RELATIVE, not absolute. Train.csv separates at
   702 mA (opening) and 470 mA (closing), but Test.csv's closing cycles have
   a median of 464 mA -- the whole distribution shifted ~37 mA and the fixed
   threshold cuts straight through the normal cluster. The Door Info Kit
   warns about exactly this ("distributions differ among doors; a uniform
   threshold leads to false alarms").

   Each cycle is scored against its OWN stream's baseline instead. On
   Train.csv the two classes separate cleanly in ratio space (normal max
   1.074, abnormal min 1.141) with one threshold covering both operations.

   Measured on five pseudo-streams cut from Train.csv:
       relative rule  100.0% accuracy, unchanged under +-9% gain or +80 mA offset
       absolute rule   91.8% under a +37 mA offset, 70.9% under a x1.09 gain
"""
import numpy as np
import pandas as pd

CURRENT = "Motor current(mA)"
CLOSING = "Door is closing"
GAP_SECONDS = 1.0
BASELINE_Q = 0.40          # firmly inside the normal cluster while abnormals < 50%
RATIO_THRESHOLD = 1.107    # midpoint of the training margin (1.074 / 1.141)


def parse_time(s: str) -> pd.Timestamp:
    """Format is Y-M-D-H-M-S-ms with no zero padding."""
    p = [int(v) for v in str(s).split("-")]
    return pd.Timestamp(p[0], p[1], p[2], p[3], p[4], p[5], p[6] * 1000)


def format_time(t: pd.Timestamp) -> str:
    return f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}-{t.microsecond//1000}"


def segment(df: pd.DataFrame) -> pd.DataFrame:
    """Cut the stream into cycles on inter-sample time gaps."""
    df = df.copy()
    df["_ts"] = df["Datetime"].map(parse_time)
    dt = df["_ts"].diff().dt.total_seconds().fillna(1e9)
    df["_seg"] = (dt > GAP_SECONDS).cumsum() - 1
    return df


def predict(path: str) -> dict:
    raw = pd.read_csv(path) if isinstance(path, str) else path
    df = segment(raw)
    g = df.groupby("_seg")

    seg = pd.DataFrame({
        "start": g["_ts"].min(),
        "end": g["_ts"].max(),
        "rows": g.size(),
        "is_close": g[CLOSING].max().astype(int),
        "cur_mean": g[CURRENT].mean(),
        "cur_max": g[CURRENT].max(),
    })

    # per-operation baseline estimated from THIS file only
    seg["baseline"] = seg.groupby("is_close")["cur_mean"].transform(
        lambda s: s.quantile(BASELINE_Q))
    seg["ratio"] = seg["cur_mean"] / seg["baseline"]
    seg["abnormal"] = seg["ratio"] >= RATIO_THRESHOLD

    out = pd.DataFrame({
        "start_time": seg["start"].map(format_time),
        "end_time": seg["end"].map(format_time),
        "prediction": np.where(seg["abnormal"], "Abnormal resistance", "Normal"),
    })

    detail = seg.assign(
        operation=np.where(seg["is_close"] == 1, "Close", "Open"),
        excess_pct=((seg["ratio"] - 1) * 100).round(1),
    )[["operation", "rows", "cur_mean", "baseline", "ratio", "excess_pct", "abnormal"]]

    return {
        "segments": out.reset_index(drop=True),
        "detail": {
            "n_segments": int(len(seg)),
            "n_abnormal": int(seg["abnormal"].sum()),
            "baselines_mA": {("Close" if k else "Open"): round(float(v), 1)
                             for k, v in seg.groupby("is_close")["baseline"].first().items()},
            "ratio_threshold": RATIO_THRESHOLD,
            "per_segment": detail.round(3).reset_index(drop=True),
            "note": "Each cycle is compared to the 40th-percentile current of the "
                    "same operation in this same file, so the rule survives "
                    "door-to-door and stream-to-stream offsets.",
        },
    }
