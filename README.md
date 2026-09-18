# NebulaX PS3 — Train Fault Prediction

Four predictors for NebulaX 2026 Problem Statement 3 (train condition monitoring).
Each one takes a single data file and returns a prediction plus a plain-language
explanation that the app can show on screen.

> **All scores below are cross-validated on the training data.** The organisers
> hold the test labels, so none of these numbers is a test result.

## Results at a glance

| Subsystem | Task | How it works | CV score | Checked with |
|---|---|---|---|---|
| **Door** | Find each open/close cycle, label it Normal or Abnormal resistance | Time-gap segmentation + motor current relative to the file's own baseline | **0.996** (110/110 on the full stream) | 5-fold × 5 seeds, 110 cycles |
| **SHM** | Estimate cumulative fatigue damage | Rainflow counting + Miner's rule with two fitted constants | **0.972** (MAPE 2.76%) | leave-one-out, 64 files |
| **ACV** | Rank the 8 cars by refrigerant-leak likelihood | Cabin temperature relative to the other cars | **0.979** (true car 1st in 5 of 6 cases, 2nd in 1) | all 6 training cases |
| **Rail** | Normal / Side I / Side II corrugation | 73 vibration features + soft-vote ensemble | **0.84 ± 0.02** macro F1 | 5-fold × 20 seeds, 272 files |

If these estimates hold, the Overall Score is roughly **0.93–0.95**. ACV (a single
test file) and Rail (only a handful of Side I files in the test set) are the two
that can move it.

## Install

```bash
pip install -r requirements.txt
```

The Rail model was saved with **scikit-learn 1.8.0**. Use that exact version so
the predictions don't drift.

## Use

```python
from ps3 import predict

predict.run("door", "Test.csv")
predict.run("shm",  "test01.csv")
predict.run("acv",  "acv_test_case.xlsx")
predict.run("rail", "Test1.csv")
```

Every call returns `{"prediction": ..., "detail": {...}}`. `detail` is written to be
shown directly in the app and explains why the prediction was made. Door is the
exception: it returns `{"segments": DataFrame, "detail": {...}}`, because one file
contains many door cycles. `predict.SUBSYSTEMS` lists each subsystem's display
name and accepted file types, which is enough to build the upload page.

## The data

| Subsystem | Files (train / test) | One file is | Used by the model | Labels |
|---|---|---|---|---|
| Door | 1 / 1 CSV stream | 50 Hz stream, 17 columns (motor current, voltage, back-EMF, commands, limit switches, door state, leaf position). Train: 18,036 rows, 110 cycles, 70 min. Test: 6,253 rows, 38 cycles | timestamps, motor current, "door is closing" flag | start, end and label per cycle; 27% abnormal |
| SHM | 64 / 16 CSV | 581,120 stress samples, one column, no header; two lines, two load conditions (AW0 / AW4) | the whole stress series | cumulative damage, 0.03–0.93 |
| ACV | 6 / 1 XLSX | 30-second telemetry for all 8 cars, 3,263–22,262 rows; 8 parameters per car (one case has 63, with different names) | indoor average temperature, setting mode | which car leaks |
| Rail | 272 / 68 CSV | 1 s at 10 kHz: 10,000 rows × 129 columns (speed pulse + vibration and shock for 64 axle boxes) | all 128 vibration/shock channels + speed | 234 Normal, 14 Side I, 24 Side II |

## Data checks and cleaning

- **Rail:** no missing values, no dead or flat channels, and no saturation (largest
  reading 35 m/s² against a ±400 m/s² range). Some shock sensors run much quieter than
  others, so the features summarise each side over many axle boxes instead of
  trusting single sensors. All 47 stationary files (< 5 km/h) are Normal and every
  fault file is at 35 km/h or faster. Speed is decoded from the 90-tooth wheel pulse.
  Wavelength features are left blank when the train is stationary: gradient boosting
  handles blanks natively, and the SVM and logistic regression fill them with the
  training median.
- **SHM:** non-finite samples would be dropped, but none occur (every file has the
  full 581,120 samples). No filtering or detrending, because the labels were made
  from the raw series.
- **Door:** the non-padded timestamp format (`Y-M-D-H-M-S-ms`) is parsed explicitly.
  Splitting on gaps longer than 1 second reproduces all 110 labelled cycles exactly.
- **ACV:** column names are read from each file's own headers, and the one case with a
  different vocabulary is mapped onto the common names. Zero temperature readings are
  treated as dropouts. Masking rows flagged "Invalid" or not in cooling mode was
  tested and gives the same ranking.

## How each model works

### Door: relative current threshold

1. **Segment:** split the stream wherever samples are more than 1 s apart. This
   matches every labelled cycle, so every IoU is 1.0 and the IoU-weighted F1 becomes
   plain accuracy.
2. **Feature:** the mean motor current of each cycle.
3. **Baseline:** for each operation (open, close), the 40th-percentile cycle current
   *in the same file*. Ratio = cycle mean ÷ baseline.
4. **Decide:** Abnormal resistance if the ratio is ≥ 1.107. That is the midpoint
   between the highest normal (1.074) and the lowest abnormal (1.141) training ratio.

The threshold is relative because current levels shift between doors. Test's closing
cycles have a median of 464 mA, right on Train's absolute cut-off of 470 mA. The Door
Info Kit warns about exactly this.

| Method | CV accuracy | +37 mA offset | ×1.09 gain | ×0.92 gain |
|---|---|---|---|---|
| **Relative threshold (used)** | **0.996** | **1.000** | **1.000** | **1.000** |
| Gradient boosting on relative features | 0.996 | 1.000 | 1.000 | 1.000 |
| Absolute threshold per operation | 0.991 | — | — | — |
| Gradient boosting on raw features | 0.984 | 0.573 | 0.545 | 0.682 |

The relative baseline is what matters, not the model type, so the simplest and most
explainable option is used. The rule stays reliable until about 55% of a file's
cycles are abnormal; beyond that the 40th percentile lands among the abnormal cycles.
Train has 27% abnormal cycles and the test predictions 17–25%. On the test file the
ratios closest to the threshold are 1.072 and 1.204, so no cycle is borderline.

### SHM: physics, not machine learning

The Info Kit states that the labels come from rainflow counting and Miner's rule, so
the formula is rebuilt rather than learned:

1. Rainflow-count the stress series. Amplitude = range ÷ 2, and half cycles count 0.5.
2. `damage = Σ count × amplitude^m / C`
3. Fit `m` and `C` on the 64 training files. For each `m` on a 0.01 grid, `C` is the
   geometric-mean ratio; keep the `m` with the lowest MAPE. Result: **m = 5.03,
   C = 7.98 × 10⁸** (stored in `params_shm.json`).

Leave-one-out MAPE is 2.76%, which scores 0.972. The fit is sharp in `m`: 2.7% at 5.0,
but 12–13% at 4.5 or 5.5. By contrast, a simple summary statistic like the standard
deviation only reaches R² 0.56, against 0.9989 for the rainflow damage sum.

Variants tested and ruled out, none of which beats 2.76%:

| Variant | Result |
|---|---|
| Range binning (15 `nbins`, 11 `binsize` settings, bin centres, rounding) | 2.56–2.61% in-sample, no real gain |
| Half cycles counted as full, or dropped | worse |
| Two-slope S-N curve (knee and both slopes fitted) | 2.81% leave-one-out |
| Goodman / Gerber mean-stress correction | 3.20% / 2.62% in-sample |
| Physics + small learned correction (std, damage shares) | 2.82–2.87% leave-one-out |

The remaining ~2.7% isn't explained by counting method, curve shape, mean stress or
file statistics. Most files are within ±3% and six are 8–15% off with no common
cause, so it looks like label noise or organiser-side preprocessing.

### ACV: warmest car relative to its neighbours

A refrigerant leak means less cooling capacity, so that car's cabin runs warmer than
the others under the same weather, time of day and route.

1. At each timestamp, each car's indoor temperature minus the median of all 8 cars.
   This cancels ambient temperature, time of day and route.
2. Average over the file to get one score per car, then rank from warmest to coolest.
3. Tiebreak: the share of time the crew switched that car to Manual Control.

`detail` reports the margin between the top two cars, with confidence high
(≥ 0.15 °C), medium (≥ 0.05 °C) or low.

On training, the true car ranks 1st in 5 of 6 cases and 2nd in 1 (score 0.979; a
random ordering scores 0.5625). Expect 0.88–1.00 on a single held-out file.

| Method | c01 | c02 | c03 | c04 | c05 | c06 | Test pick (margin) |
|---|---|---|---|---|---|---|---|
| **Cabin temp vs car median (used)** | 1 | 1 | 1 | 2 | 1 | 1 | **Car 01** (+0.035 °C) |
| Valid, cooling-mode rows only | 1 | 1 | 1 | n/a | 1 | 1 | Car 01 (+0.042) |
| Cabin temp vs each car's own setpoint | 1 | 1 | 1 | 2 | 1 | 1 | Car 01 (+0.166) |
| Warmest half of timestamps only | 1 | 1 | 1 | 3 | 1 | 1 | Car 01 (+0.016) |
| Share of time in Full Cooling | 2 | 2 | 3 | n/a | 6 | 7 | no signal |
| Share of time load-halved | 1 | 2 | 3 | n/a | 4 | 6 | no signal |

(Numbers are the rank of the true faulty car; 1 = correct.) On the test file every
temperature variant picks Car 01. Relative to each car's own setpoint, the margin is
almost 5× larger, which suggests the runner-up is warmer because of a higher setpoint,
not a leak. Caveat: the method was compared against other candidates on the same six
cases, which is selection on the evaluation set. It is trusted because it was the
physics prediction going in, not the winner of a search.

### Rail: vibration features + soft-vote ensemble

Each car has 8 axle boxes. Odd positions ride on the Side I rail and even positions
on Side II, and each box carries a vibration and a shock sensor.

**Features (73), computed per file by `rail.featurize` (~1 s per file):**

- **Speed**, from the 0/1 transitions of the 90-tooth wheel pulse and the 0.85 m wheel.
- **Side features (56):** for each of 4 channel groups (Side I vibration, Side II
  vibration, Side I shock, Side II shock):
  - mean and max channel RMS, kurtosis and crest factor;
  - energy share in 6 frequency bands (50–5,000 Hz);
  - energy share in 4 wavelength bands (20–320 mm, using wavelength = speed ÷ frequency).
- **Loudest-axle-box features (16):** per side and sensor type, the log-RMS of the
  loudest box, the mean of the 3 loudest, the median box, and loudest minus the file
  median.

**Model:** a soft vote (average of class probabilities) of three different learners.
All three use balanced class weights to account for the rare fault classes.

- gradient boosting (HistGradientBoosting, 300 iterations);
- RBF-SVM (C = 3, standardised);
- logistic regression (C = 0.1, standardised).

| | Previous model (gradient boosting, 57 features) | Current model (ensemble, 73 features) |
|---|---|---|
| Macro F1, 5-fold × 20 seeds | 0.80 ± 0.04 | **0.84 ± 0.02** (better on 18 of 20 splits) |
| Normal F1 | 0.978 | 0.978 |
| Side I F1 | 0.597 | **0.676** |
| Side II F1 | 0.830 | **0.863** |

**What the data showed:**

- **Loudest boxes carry the signal.** Corrugation shows up in the few loudest axle
  boxes on the faulty side. Median-based features wash it out: left-vs-right
  contrasts built on medians scored only 0.46.
- **The sides aren't mirror images.** Swapping Side I and Side II channels to create
  extra examples made things worse (0.805).
- **Errors are mostly misses.** Most Side I errors are faults called Normal, not
  confusion with Side II.
- **Faults need speed.** Faults only appear at 35 km/h or more, and stationary files
  are always Normal.

**Other approaches tested (macro F1):**

| Approach | Macro F1 |
|---|---|
| Random forest / extra trees | 0.68 / 0.72 |
| SVM alone / logistic regression alone | 0.76 / 0.76 |
| Gradient boosting alone | 0.81 |
| Per-sensor speed calibration | 0.70–0.74 |
| Extra per-band loudest-box features | 0.78 |
| Symmetric "is this side corrugated?" scorer | 0.42–0.79 |
| Decision rule tuned inside the folds | +0.012, not adopted |

**Caveats:**

- The ensemble was chosen from about 15 configurations, so 0.84 is slightly
  optimistic. It was confirmed on 10 fresh random splits, beating the previous model
  on all 10.
- With 3–4 Side I files expected in the test set, the realised score can move by
  about 0.1 either way.
- One test file is a genuine toss-up: Test9 is 65% Side II under the old model and
  65% Normal under the new one.

## Validation and honesty notes

- Every score here comes from cross-validation on training data.
- Everything that is fitted (the Door threshold, the SHM constants, the Rail features
  and model) is fitted inside the training folds only.
- Door's relative baseline uses only the test file's own unlabelled readings, which
  are available at prediction time.
- Selection effects are stated where they exist: ACV was compared against other
  candidates on its 6 cases, and the Rail ensemble was picked from ~15 configurations.
- Rail has a speed confound: in training, faults only occur above 35 km/h. A slow
  corrugated section in real service would likely be called Normal.

## Suggested improvements

1. **Pin `scikit-learn==1.8.0`** in `requirements.txt`, since the Rail model was saved
   with it.
2. **Rail: time-localised features.** In a 1-second window, corrugation may sit under
   only a few wheels. Loudness peaks or impact counts in short windows (50–100 ms) per
   axle box could sharpen the signal.
3. **Rail: axle-sequence consistency.** A corrugated stretch passes under each axle on
   one side in turn, delayed by axle spacing ÷ speed. Cross-correlating band-passed
   envelopes along a side could separate real corrugation from one noisy sensor.
4. **Rail: show uncertainty in the app.** Flag files whose top class probability is
   below ~0.7 (such as Test9) as "needs inspection" instead of giving a hard label.
5. **ACV: show the setpoint-adjusted margin** as supporting evidence in the
   explanation, and apply validity and cooling-mode masks when those columns exist.
6. **Door: add a sanity check in the app.** Warn when the current ratios show no clear
   gap, or when more than half of one operation's cycles look abnormal, because that
   is where the 40th-percentile baseline breaks.
7. **SHM: ask the organisers for their exact rainflow and S-N settings.** The remaining
   2.7% looks like label-side noise. Reporting remaining life (1 − damage) would make
   the output more useful for maintenance planning.
8. **More data beats more modelling.** Extra Side I files and more ACV cases would
   improve reliability more than any model change.

## Retraining

Door and ACV need no training.

```bash
python fit_shm.py  <SHM/Train dir>  <SHM/Train_Labels.csv>
python fit_rail.py <Rail_Corrugation/Train dir> <Rail_Corrugation/Train_Labels.csv>
```

`fit_rail.py` takes a few minutes for 272 files and caches features to
`rail_feats_v2.pkl`. It prints the cross-validated macro F1 and saves
`ps3/model_rail.joblib`.

## Generating submissions

```bash
python make_submissions.py /path/to/02_Datasets
```

This writes the four CSVs to `submission/`. It is for checking the output format only.
The real submission should be produced by uploading files through the deployed app, so
the predictions demonstrably come from the app itself.

## Repository layout

```
ps3/
    __init__.py
    predict.py          entry point: predict.run(subsystem, path)
    door.py  shm.py  acv.py  rail.py
    params_shm.json     fitted m and C
    model_rail.joblib   the only trained model (soft-vote ensemble)
fit_shm.py              refits m and C
fit_rail.py             refits the Rail ensemble
make_submissions.py     offline submission generator (format check)
requirements.txt
submission/             four formatted CSVs
```
