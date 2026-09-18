"""Fits the SHM models on the training files:
  1. Miner's-rule constants m and C -> ps3/params_shm.json (used as a cross-check
     and for the explanation shown in the app);
  2. the prediction model, ridge regression on rainflow damage sums + signal
     statistics -> ps3/model_shm.joblib.
Prints leave-one-out MAPE for both (64 files: 2.76% formula, 2.31% model)."""
import json, sys, os
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ps3 import shm

def main(train_dir, labels_csv, out=None):
    lab = pd.read_csv(labels_csv)
    series = [shm.load_series(os.path.join(train_dir, f)) for f in lab["filename"]]
    cyc = [shm.cycles(x) for x in series]
    y = lab["damage"].to_numpy()

    # 1. Miner's-rule formula
    ms = np.arange(3.0, 8.001, 0.01)
    logS = np.log(np.array([[shm.damage_sum(c, m) for c in cyc] for m in ms]))
    logC = (logS - np.log(y)).mean(1)
    mape = (np.abs(y - np.exp(logS - logC[:, None])) / y).mean(1)
    k = int(mape.argmin())
    p = {"m": round(float(ms[k]), 3), "C": float(np.exp(logC[k])),
         "n_train_files": len(y), "in_sample_mape": round(float(mape[k]), 5)}
    out = out or shm.PARAMS_PATH
    json.dump(p, open(out, "w"), indent=1)
    print(f"formula: m={p['m']}  C={p['C']:.4g}  in-sample MAPE={p['in_sample_mape']:.2%}  -> {out}")

    # 2. ridge regression, validated leave-one-out
    F = pd.DataFrame([shm.features(x, c) for x, c in zip(series, cyc)])
    X = F[sorted(F.columns)]
    ly = np.log(y)
    pred = np.empty(len(y))
    for j in range(len(y)):
        tr = np.delete(np.arange(len(y)), j)
        pred[j] = np.exp(shm.make_model().fit(X.iloc[tr], ly[tr]).predict(X.iloc[[j]])[0])
    print(f"model: leave-one-out MAPE={np.mean(np.abs(pred - y) / y):.2%}")
    model = shm.make_model().fit(X, ly)
    fit = np.exp(model.predict(X))
    print(f"model: in-sample MAPE={np.mean(np.abs(fit - y) / y):.2%}")
    joblib.dump({"model": model, "features": list(X.columns)}, shm.MODEL_PATH)
    print("saved", shm.MODEL_PATH)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
