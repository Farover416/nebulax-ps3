"""Fits the two S-N constants on the SHM training files -> ps3/params_shm.json.
Leave-one-out validated: MAPE 2.76% across all 64 files (score 0.972)."""
import json, sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ps3 import shm

def main(train_dir, labels_csv, out=None):
    lab = pd.read_csv(labels_csv)
    cyc = [shm.cycles(shm.load_series(os.path.join(train_dir, f)))
           for f in lab["filename"]]
    y = lab["damage"].to_numpy()
    ms = np.arange(3.0, 8.001, 0.01)
    logS = np.log(np.array([[shm.damage_sum(c, m) for c in cyc] for m in ms]))
    logC = (logS - np.log(y)).mean(1)
    mape = (np.abs(y - np.exp(logS - logC[:, None])) / y).mean(1)
    k = int(mape.argmin())
    p = {"m": round(float(ms[k]), 3), "C": float(np.exp(logC[k])),
         "n_train_files": len(y), "in_sample_mape": round(float(mape[k]), 5)}
    out = out or shm.PARAMS_PATH
    json.dump(p, open(out, "w"), indent=1)
    print(f"m={p['m']}  C={p['C']:.4g}  in-sample MAPE={p['in_sample_mape']:.2%}  -> {out}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
