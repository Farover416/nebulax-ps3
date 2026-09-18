"""Runs every subsystem over the released test inputs and writes the four
submission CSVs in the exact example format.

    python make_submissions.py /path/to/02_Datasets [outdir]

The real submission should be produced by POSTing these same files to the
deployed endpoint, so that predictions.zip demonstrably comes from the app.
This script is the offline equivalent for checking output shape.
"""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ps3 import predict


def main(root, outdir="submission"):
    os.makedirs(outdir, exist_ok=True)

    rows = [{"file_id": os.path.basename(f),
             "prediction": predict.shm(f)["prediction"]}
            for f in sorted(glob.glob(f"{root}/SHM/Test/*.csv"))]
    pd.DataFrame(rows).to_csv(f"{outdir}/shm_predictions.csv", index=False)
    print(f"shm  : {len(rows)} files")

    f = f"{root}/Door/Test.csv"
    seg = predict.door(f)["segments"]
    seg.to_csv(f"{outdir}/door_predictions.csv", index=False)
    print(f"door : {len(seg)} segments, "
          f"{int((seg.prediction != 'Normal').sum())} abnormal")

    rows = [{"file_id": os.path.basename(f), "ranked_cars": predict.acv(f)["ranked_cars"]}
            for f in sorted(glob.glob(f"{root}/ACV/Test/*.xlsx"))]
    pd.DataFrame(rows).to_csv(f"{outdir}/acv_predictions.csv", index=False)
    print(f"acv  : {len(rows)} files")

    rows = [{"file_id": os.path.basename(f), "prediction": predict.rail(f)["prediction"]}
            for f in sorted(glob.glob(f"{root}/Rail_Corrugation/Test/*.csv"))]
    pd.DataFrame(rows).to_csv(f"{outdir}/rail_predictions.csv", index=False)
    print(f"rail : {len(rows)} files -> "
          f"{pd.Series([r['prediction'] for r in rows]).value_counts().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "submission")
