# NebulaX PS3 — Train Fault Prediction

Four predictors for NebulaX 2026 Problem Statement 3. Each one takes a data
file and returns a prediction plus a plain-language explanation.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from ps3 import predict

predict.run("door", "Test.csv")
predict.run("shm",  "test01.csv")
predict.run("acv",  "acv_test_case.xlsx")
predict.run("rail", "Test1.csv")
```

Every call returns a dictionary shaped like this:

```python
{"prediction": ..., "detail": {...}}
```

`detail` is written to be shown directly on screen in the app. It explains why
the prediction was made.

Door is the exception: it returns `{"segments": DataFrame, "detail": {...}}`,
because one file contains many door cycles.

`predict.SUBSYSTEMS` lists each subsystem's accepted file types and a display
name, which is enough to build the upload page.

## The four subsystems

| Key | What it does | Input | Output |
|---|---|---|---|
| `door` | Finds each door open/close cycle and labels it | one long CSV | one row per cycle: start, end, Normal or Abnormal resistance |
| `shm` | Estimates accumulated metal fatigue | one long CSV | a number between 0 and 1 |
| `acv` | Finds which car has an aircon refrigerant leak | one XLSX | all 8 cars ranked, most likely first |
| `rail` | Detects track corrugation from vibration | one CSV | Normal, Side I, or Side II |

## How each one works

**SHM** is not machine learning. The labels were made using rainflow cycle
counting and Miner's rule, so the original formula is recovered directly:
`damage = sum(cycles x amplitude^m) / C`, with `m = 5.03` and `C = 7.98e8`.
Leave-one-out error across all 64 training files is 2.76%.

**Door** splits the stream wherever there is a time gap longer than 1 second.
This reproduces all 110 training segments exactly. Each cycle is then compared
to the 40th-percentile motor current of the same operation *in the same file*.
The comparison is relative, not a fixed milliamp threshold, because current
levels shift between doors — a fixed threshold scored 71% under a 9% shift,
the relative one scored 100%.

**ACV** takes each car's cabin temperature minus the median across all 8 cars
at that moment, then averages it. A refrigerant leak means less cooling, so
that car runs warmer than its neighbours. Ranks the correct car first in 5 of
6 training cases.

**Rail** extracts vibration statistics and frequency-band energy for each side
of the train, then classifies with a gradient-boosted tree. Side I is
positions 1, 3, 5, 7 and Side II is positions 2, 4, 6, 8.

## Measured scores

All cross-validated on training data only.

| Subsystem | Score | How it was checked |
|---|---|---|
| Door | 0.991 | 5-fold CV, 110 segments |
| SHM | 0.972 | leave-one-out, 64 files |
| ACV | ~0.90 | 6 of 6 training cases |
| Rail | 0.81 +/- 0.03 | 5-fold CV across 5 seeds, 272 files |

Rail is the weak one. Its Side I class has only 14 training examples and an
F1 of 0.538, so the real score can move by about 0.1 either way depending on
which files land in the test set.

## Retraining

Only Rail has a trained model. The other three need no training.

```bash
python fit_shm.py  <SHM/Train dir>  <SHM/Train_Labels.csv>
python fit_rail.py <Rail/Train dir> <Rail/Train_Labels.csv>
```

`fit_rail.py` takes about 3 minutes for 272 files and caches features to
`rail_feats.pkl`.

## Generating submissions

```bash
python make_submissions.py /path/to/02_Datasets
```

This writes the four CSVs to `submission/`. It is for checking the output
format only. The real submission should be produced by uploading files through
the deployed app, so the predictions demonstrably come from the app itself.
