"""SHM: cumulative fatigue damage from a dynamic stress time series.

Model: ridge regression on log damage, trained on the 64 labelled files.
Features (30), computed per file:
  - 9 rainflow damage sums  log( sum_i n_i * amp_i^m )  for m = 3.0, 3.5 ... 7.0,
    so the model learns the S-N weighting from the data instead of it being fixed;
  - 21 signal statistics (mean, spread, shape, range, percentiles, step size,
    turning-point rate, spectral band shares).

Leave-one-out MAPE across all 64 files: 2.31% (score 0.977), against 2.76% for
the fitted Miner's-rule formula D = sum(n * amp^m) / C (m = 5.03, C = 7.98e8)
and 10-20% for machine learning on the signal statistics alone. The formula is
still computed for every file and shown in `detail` as a cross-check.
"""
import json
import os

import numpy as np
import pandas as pd

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "params_shm.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_shm.joblib")
DAMAGE_EXPONENTS = np.arange(3.0, 7.01, 0.5)
_P = None
_M = None


def _params() -> dict:
    global _P
    if _P is None:
        with open(PARAMS_PATH) as f:
            _P = json.load(f)
    return _P


def _model() -> dict:
    global _M
    if _M is None:
        import joblib
        _M = joblib.load(MODEL_PATH)
    return _M


def load_series(path: str) -> np.ndarray:
    """Files are headerless single-column CSVs, 581,120 samples each."""
    s = pd.read_csv(path, header=None, dtype=np.float64).iloc[:, 0].to_numpy()
    return s[np.isfinite(s)]


def cycles(x: np.ndarray) -> np.ndarray:
    """Rainflow cycle extraction -> array of (amplitude, count)."""
    import rainflow
    return np.array([[rng / 2.0, cnt] for rng, mean, cnt, i, j
                     in rainflow.extract_cycles(x)])


def damage_sum(c: np.ndarray, m: float) -> float:
    return float(np.sum(c[:, 0] ** m * c[:, 1]))


def features(x: np.ndarray, c: np.ndarray) -> pd.Series:
    """The 30 model inputs for one stress series and its rainflow cycles."""
    from scipy import signal, stats
    out = {f"rf_logdmg_m{m:.1f}": np.log(damage_sum(c, m)) for m in DAMAGE_EXPONENTS}
    d = np.diff(x)
    z = x - x.mean()
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    fr, psd = signal.welch(z, nperseg=4096)
    edges = np.quantile(np.arange(len(fr)), [0, .02, .05, .1, .2, .4, 1]).astype(int)
    out.update({
        "st_log_std": np.log(x.std()), "st_mean": x.mean(), "st_skew": stats.skew(x),
        "st_kurt": stats.kurtosis(x), "st_log_range": np.log(x.max() - x.min()),
        **{f"st_p{p:02d}": v for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)},
        "st_log_absdiff": np.log(np.abs(d).mean()), "st_log_diffstd": np.log(d.std()),
        "st_turning_rate": np.sum(np.diff(np.sign(d)) != 0) / len(x),
        **{f"st_band{i}": np.log(psd[a:b].sum() / psd.sum() + 1e-12)
           for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))},
    })
    return pd.Series(out)


def featurize(path: str) -> pd.Series:
    x = load_series(path)
    return features(x, cycles(x))


def make_model():
    """Standardised features -> ridge regression on log(damage); the
    regularisation strength is chosen by internal cross-validation."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 2, 13)))


def predict(path: str) -> dict:
    p = _params()
    x = load_series(path)
    c = cycles(x)
    M = _model()
    X = features(x, c).reindex(M["features"]).to_frame().T
    D = float(np.exp(M["model"].predict(X)[0]))
    D_formula = damage_sum(c, p["m"]) / p["C"]

    # explanation: which amplitude bands drive the damage
    edges = np.array([0, 5, 10, 15, 20, 25, 30, np.inf])
    contrib = c[:, 0] ** p["m"] * c[:, 1]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (c[:, 0] >= lo) & (c[:, 0] < hi)
        if sel.any():
            bands.append({
                "amplitude_range": f"{lo:.0f}-{'inf' if np.isinf(hi) else f'{hi:.0f}'}",
                "cycles": float(c[sel, 1].sum()),
                "share_of_damage": round(float(contrib[sel].sum() / contrib.sum()), 4),
            })
    bands.sort(key=lambda b: -b["share_of_damage"])

    return {
        "prediction": round(D, 6),
        "detail": {
            "model": "ridge regression on rainflow damage sums + signal statistics",
            "miners_rule_estimate": round(float(D_formula), 6),
            "model_vs_formula_pct": round(100 * (D / D_formula - 1), 2),
            "samples": int(len(x)),
            "rainflow_cycles": int(c[:, 1].sum()),
            "largest_amplitude": round(float(c[:, 0].max()), 3),
            "sn_exponent_m": p["m"],
            "sn_constant_C": p["C"],
            "damage_by_amplitude_band": bands[:4],
            "note": "Damage is dominated by the few largest-amplitude cycles "
                    f"because stress enters at roughly the power of {p['m']:.2f}.",
        },
    }
