"""Extracts features from the Rail training files and fits the classifier
-> ps3/model_rail.joblib.  ~3 min for 272 files."""
import sys, os, time
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ps3 import rail
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

def main(train_dir, labels_csv, cache="rail_feats.pkl"):
    lab = pd.read_csv(labels_csv)
    if os.path.exists(cache):
        F = pd.read_pickle(cache); print(f"loaded cached features {F.shape}")
    else:
        t0 = time.time(); rows = []
        for i, r in lab.iterrows():
            rows.append(rail.featurize(os.path.join(train_dir, r.filename)))
            if i % 40 == 0: print(f"  {i}/{len(lab)}  {time.time()-t0:.0f}s", flush=True)
        F = pd.DataFrame(rows); F.to_pickle(cache)
    X = F.drop(columns=[c for c in ("filename", "label") if c in F.columns])
    X = X[sorted(X.columns)]
    y = lab["label"].to_numpy()
    m = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0)
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"\nCV macro F1 = {f1_score(y, p, average='macro'):.4f}")
    print(classification_report(y, p, digits=3))
    m.fit(X, y)
    joblib.dump({"model": m, "features": list(X.columns)}, rail.MODEL_PATH)
    print("saved", rail.MODEL_PATH)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
