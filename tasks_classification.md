# TCRP Classification Experiments

Extension of TCRP from forecasting to time-series classification.
Theory is identical; only the decoder output and loss change.
All concept families, alignment loss, analysis pass, and conservation
theorem carry over without modification.

---

## Architectural delta from forecasting

Only two changes relative to `tcrp_v3`:

```python
# Decoder: C classes instead of H horizons
decoder = nn.Linear(K, C)           # was nn.Linear(K, H)

# Loss: cross-entropy instead of MSE
L_forecast = F.cross_entropy(y_hat, y_true)

# Relevance initialisation: predicted class score
R_out = y_hat[k_star]               # was y_hat[h_star]
```

Everything else — segmentation, encoder, bottleneck, alignment,
pooling, GRL, analysis pass — is unchanged. The conservation theorem
holds: sum*t R_x_t = f*{k_star}(x) = predicted class logit.

Implement as `TCRPClassifier(config: TCRPClassConfig)` wrapping the
existing `TCRPForecaster` components with the two substitutions above.

```python
@dataclass
class TCRPClassConfig:
    T: int               # series length
    C: int               # number of classes
    L: int  = 20         # segment window
    stride: int = 5
    d: int  = 64
    K: int  = 22
    periods: list = field(default_factory=lambda: [])  # dataset-dependent
    alpha: float = 5.0
    beta:  float = 5.0
    lambda1: float = 0.1
    lambda2: float = 1e-4
    adversarial: bool = False
    alpha_max:   float = 1.0
    warmup_epochs: int = 20
```

---

## Experiment set 1 — Physiological / Clinical

### EXP-C01 · ECG arrhythmia classification (ECG5000)

**Dataset:** ECG5000 \[Chen et al., 2015\]

- 5,000 ECG segments, length T=140, 5 classes
  (normal sinus, R-on-T PVC, PVC, paced beat, others)
- Source: UCR Time Series Archive
- Download: `http://www.timeseriesclassification.com/description.php?Dataset=ECG5000`

**Configuration:**

- L=20, stride=5, C=5
- Candidate periods: {4, 6, 8, 10} (samples per beat at typical heart rate)
- K=22 (full concept vocabulary; periodicity at heartbeat scale is relevant)

**Baselines:**

- InceptionTime \[Ismail Fawaz et al., 2020\]
- ROCKET \[Dempster et al., 2020\]
- ResNet-1D \[Wang et al., 2017\]
- PatchTST adapted for classification
- TCAV \[Kim et al., 2018\] applied to ResNet-1D (post-hoc concept baseline)
- WinIT \[Rooke et al., 2021\] applied to ResNet-1D (post-hoc attribution baseline)

**Primary metrics:**

- Classification accuracy, macro F1 (class imbalance present)
- CAS per concept family against cardiologist annotations
  (annotation protocol: label each segment as dominated by
  monotonicity / periodicity / stochasticity / shape anomaly)
- Temporal faithfulness (comprehensiveness + sufficiency at p=20%)

**Key expected findings:**

- Periodicity concept dominates for normal sinus (regular R-R interval)
- Stochasticity concept dominates for atrial fibrillation (irregular R-R)
- Shape concepts (skewness, jump indicator) dominate for PVC (ectopic beat)
- TCRP concept maps should spatially localise to P-wave, QRS complex,
  T-wave regions — verifiable against cardiologist ground truth

**Expert annotation protocol:**

- 200 held-out segments labelled by one cardiologist
- Label = primary concept driving their classification decision
- Classes: {0=periodicity, 1=stochasticity, 2=monotonicity, 3=shape}
- Inter-rater reliability measured on 50-segment overlap with second annotator

---

### EXP-C02 · ECG heartbeat classification (MIT-BIH)

**Dataset:** MIT-BIH Arrhythmia Database \[Moody & Mark, 2001\]

- 48 half-hour ECG recordings, sampled at 360 Hz
- 5 AAMI classes: N (normal), S (supraventricular), V (ventricular),
  F (fusion), Q (unknown)
- Standard 70/30 patient-level split (no patient appears in both splits)
- Preprocessing: extract 187-sample heartbeat windows centred on R-peak

**Configuration:**

- T=187, L=25, stride=5, C=5
- Periods: {18, 36} (half-beat and full-beat at 360Hz / ~100bpm)

**Baselines:** same as EXP-C01 plus:

- CNN-LSTM \[Yildirim et al., 2018\]
- Transformer-based ECG classifier \[Natarajan et al., 2020\]

**Clinical evaluation:**

- Compare TCRP concept maps against known clinical markers:
  - V class: wide QRS → high monotonicity, high jump indicator
  - S class: narrow QRS, abnormal P wave → stochasticity in P-wave region
  - F class: mixed morphology → high break score at QRS onset
- Report which concept family achieves highest CAS per AAMI class

---

### EXP-C03 · Sleep stage classification (Sleep-EDF)

**Dataset:** Sleep-EDF Cassette \[Kemp et al., 2000; Physionet\]

- 20 subjects, 2-channel EEG + EOG, sampled at 100 Hz
- 5 sleep stages: Wake, N1, N2, N3, REM
- 30-second epochs (3000 samples each)
- Use single Fpz-Cz EEG channel for univariate setting

**Configuration:**

- T=3000, L=100, stride=50, C=5
- Periods: {100, 500} (1s and 5s physiological oscillations)
- K=22; periodicity especially important (sleep spindles at ~14Hz,
  delta waves at 0.5–4Hz captured by spectral concept)

**Baselines:**

- SERF \[Eldele et al., 2021\]
- AttnSleep \[Eldele et al., 2021\]
- TimesNet adapted for classification

**Unique evaluation:**

- TCRP concept maps should localise sleep spindles (periodic bursts,
  high periodicity concept) in N2 stage and delta waves (high
  monotonicity + low stochasticity) in N3 stage
- Report concept relevance profile per sleep stage:
  expected pattern: N3 dominated by monotonicity + low stochasticity;
  REM dominated by stochasticity; N2 by periodicity

---

## Experiment set 2 — Industrial / Fault Detection

### EXP-C04 · Bearing fault classification (CWRU)

**Dataset:** Case Western Reserve University Bearing Dataset \[Smith & Randall, 2015\]

- Vibration signals at 12kHz, 4 fault conditions × 4 severity levels
- Classes: Normal, Inner race fault, Outer race fault, Ball fault
- Standard split: 70% train / 30% test, stratified by severity

**Configuration:**

- T=1024, L=64, stride=16, C=4
- Periods: {32, 64} (fault frequencies at typical shaft speeds)
- Shape concepts especially relevant: faults manifest as periodic
  impulses (high jump indicator, high kurtosis)

**Baselines:**

- 1D-CNN \[Janssens et al., 2016\]
- Deep Convolutional Neural Network \[Wen et al., 2018\]
- WDCNN \[Zhang et al., 2017\]
- InceptionTime

**Domain-specific evaluation:**

- Kurtosis concept should receive highest relevance for all fault classes
  (impulsive vibration is the diagnostic signature of bearing faults)
- Jump indicator should activate at fault frequency intervals
- Periodicity map should align with shaft rotation frequency
- Report: concept relevance decomposition per fault type and severity level
- Check: does fault severity correlate with magnitude of shape concept relevance?

---

### EXP-C05 · HAR — Human activity recognition (UCI-HAR)

**Dataset:** UCI Human Activity Recognition \[Anguita et al., 2013\]

- 30 subjects, accelerometer + gyroscope at 50Hz
- 6 activities: walking, walking upstairs, walking downstairs,
  sitting, standing, lying
- 2.56-second windows (128 samples), 50% overlap
- Use total acceleration magnitude (univariate) or all 6 channels

**Configuration (univariate):**

- T=128, L=20, stride=5, C=6
- Periods: {10, 20} (step frequency at ~2Hz and double-step)

**Configuration (multivariate):**

- Apply concept scorer per channel, pool across channels
- Cross-channel co-monotonicity concept for coordinated limb movement

**Baselines:**

- DeepConvLSTM \[Ordóñez & Roggen, 2016\]
- Attend and Discriminate \[Abedin et al., 2021\]
- ROCKET

**Expected concept patterns:**

- Walking: strong periodicity (regular step cadence)
- Sitting/standing: near-zero monotonicity, low stochasticity (static)
- Stair climbing: strong monotonicity + high volatility (irregular steps)
- Lying: near-zero all concepts (no movement signal)
- Report: does the periodicity concept at the step frequency distinguish
  walking from static activities better than raw gradient saliency?

---

### EXP-C06 · EthanolConcentration (UCR)

**Dataset:** EthanolConcentration \[UCR Archive\]

- 524 time series, length T=1751, 4 classes (ethanol concentration levels)
- Spectrometer readings from food adulteration detection
- Multivariate: 3 channels

**Configuration:**

- T=1751, L=50, stride=20, C=4
- Periods: {} (no strong periodicity expected; disable periodicity concepts)
- K=18 (full vocabulary minus periodicity)

**Why interesting for TCRP:**

- No obvious periodic structure → tests whether stochasticity and
  autocorrelation concepts carry the classification signal
- Ground truth: spectral absorption features at specific wavelengths;
  the periodicity concept should receive near-zero relevance (certified)
- Tests T-31 orthogonal bypass: if periodicity concepts receive high
  relevance despite being analytically near-zero, the bottleneck is
  leaking non-concept information

---

## Experiment set 3 — Financial / Economic

### EXP-C07 · Regime classification (S&P 500)

**Dataset:** S&P 500 daily log-returns 2000–2022, labelled with
NBER recession indicators (binary: recession / expansion) and
VIX-based volatility regimes (low / medium / high VIX tercile → 3 classes)

**Two tasks:**

- Task A: binary recession classification (C=2)
- Task B: 3-class volatility regime classification (C=3)

**Configuration:**

- T=252 (trading year look-back), L=21 (trading month), stride=5, C=2 or 3
- No periodicity concepts (returns near-white-noise at segment scale)
- K=18

**Baselines:**

- Logistic regression on rolling Sharpe / volatility features
- ROCKET
- TimeSHAP applied to LSTM classifier

**Expected findings:**

- Task A: monotonicity and observed tendency dominate recession detection
  (sustained negative trend is the defining recession signal)
- Task B: volatility level and ARCH clustering dominate VIX regime
  classification
- CAS evaluation: compare against risk manager annotations on
  100 held-out windows labelled by primary concept driving their
  regime assessment

---

### EXP-C08 · Trend / mean-reversion regime classification (FX)

**Dataset:** Daily log-returns of 8 currency pairs vs USD, 1990–2016
\[Lai et al., 2018\]

**Task:** classify each 21-day window as:

- Class 0: trending (|Hurst - 0.5| > 0.15, H > 0.5)
- Class 1: mean-reverting (|Hurst - 0.5| > 0.15, H < 0.5)
- Class 2: random walk (|Hurst - 0.5| ≤ 0.15)

Labels derived from rolling Hurst exponent estimation — no human
annotation required.

**Why interesting:**

- Ground truth concept labels are analytically derived from the same
  Hurst estimator used inside the stochasticity concept $\xi$
- TCRP should achieve near-perfect CAS on the stochasticity concept
  family for this task — a sanity check that the bottleneck is learning
  what it claims to learn
- If CAS on stochasticity < 0.8 for this task, something is wrong with
  either the concept scorer or the alignment training

---

## Experiment set 4 — UCR/UEA Benchmark (Breadth)

Run TCRP on a representative subset of the UCR Time Series
Classification Archive to measure breadth of concept coverage.
Select 15 datasets spanning different domains, lengths, and class counts.

### EXP-C09 · UCR benchmark suite

**Selected datasets (representative, not exhaustive):**

| Dataset                   | T    | C   | Domain         | Key concept expected       |
| ------------------------- | ---- | --- | -------------- | -------------------------- |
| FaceDetection             | 62   | 2   | EEG/MEG        | Periodicity, stochasticity |
| EthanolConcentration      | 1751 | 4   | Spectrometer   | Autocorrelation            |
| MotorImagery              | 3000 | 2   | EEG            | Stochasticity, shape       |
| SelfRegulationSCP1        | 896  | 2   | EEG            | Monotonicity, breaks       |
| Heartbeat                 | 405  | 2   | PCG/ECG        | Periodicity, shape         |
| ArticularyWordRecognition | 144  | 25  | Motion capture | Trend, curvature           |
| AtrialFibrillation        | 640  | 3   | ECG            | Stochasticity, periodicity |
| BasicMotions              | 100  | 4   | IMU            | Monotonicity, volatility   |
| Epilepsy                  | 206  | 4   | Accelerometer  | Breaks, stochasticity      |
| NATOPS                    | 51   | 6   | Aircraft       | Curvature, tendency        |
| PhonemeSpectra            | 217  | 39  | Audio          | Periodicity, shape         |
| PEMS-BAY                  | 52   | 7   | Traffic        | Periodicity, breaks        |
| RacketSports              | 30   | 4   | Motion         | Trend, volatility          |
| StandWalkJump             | 2500 | 3   | Force plate    | Breaks, jump               |
| UWaveGestureLibrary       | 315  | 8   | Accelerometer  | Trend, curvature           |

**Evaluation protocol:**

- 5 random seeds, report mean ± std accuracy
- Compare against: ROCKET, InceptionTime, Hydra \[Dempster et al., 2023\]
- Report bypass ratio (T-31) per dataset — datasets where bypass > 0.3
  flag concept vocabulary gaps
- Report dominant concept per class per dataset (argmax of mean
  $|\mathcal{R}^{(\h)}_k|$ across test samples per class)

**Aggregate analysis:**

- Cluster datasets by dominant concept profile — do datasets where
  periodicity dominates cluster together by domain?
- Identify datasets where all concepts have low relevance — these are
  candidates for concept vocabulary extension

---

## Experiment set 5 — Ablations

### EXP-C10 · Concept family ablation (which families matter?)

For EXP-C01 (ECG5000) and EXP-C04 (CWRU):

- Train TCRP with each concept family individually removed (K reduced accordingly)
- Train TCRP with only one concept family active (all others set to zero)
- Report accuracy and CAS for each configuration

Expected result: for ECG5000, removing periodicity degrades accuracy
most; for CWRU, removing shape (kurtosis, jump indicator) degrades most.

---

### EXP-C11 · Adversarial training ablation (T\* regime)

For EXP-C01 and EXP-C07:

- Standard TCRP vs adversarial TCRP (T\*-04 trainer)
- Report: accuracy, CAS, concept purity score (T\*-05), bypass ratio (T-31)
- If accuracy is within 1%, report purity improvement as the primary gain
- Plot concept purity trajectories over training epochs for both modes

---

### EXP-C12 · Context depth ablation

For EXP-C01 and EXP-C04:

- P\* = L (context-free, default)
- P\* = 1.5L (moderate context extension)
- P\* = 2L (full context)
- Measure: accuracy, CAS, and specifically CAS on break and tendency
  concepts (most context-sensitive)

---

### EXP-C13 · Segment length ablation

For EXP-C01 (T=140):

- L ∈ {10, 15, 20, 25, 30, 40}
- Report accuracy and CAS per L
- Expected: CAS peaks at L ∈ [15, 25]; accuracy less sensitive

---

## Benchmark positioning

### Primary claim

TCRP matches or exceeds state-of-the-art classification accuracy while
providing concept-level explanations that align substantially better with
domain-expert annotations than post-hoc attribution methods.

### Required results to support claim

| Result                            | Datasets          | Threshold           |
| --------------------------------- | ----------------- | ------------------- | ----------------- | ------ |
| Accuracy ≥ InceptionTime          | EXP-C01–C08       | All 8 datasets      |
| Accuracy ≥ ROCKET                 | EXP-C09           | ≥ 10 of 15 datasets |
| CAS > TCAV by ≥ 0.15              | EXP-C01, C04      | Both datasets       |
| CAS > gradient saliency by ≥ 0.25 | EXP-C01, C04, C07 | All three           |
| Bypass ratio < 0.3                | EXP-C01–C08       | All 8 datasets      |
| Conservation holds (Theorem 1)    | All               |                     | R_x.sum - f_kstar | < 1e-4 |

### Failure modes to report honestly

- Datasets where TCRP underperforms ROCKET by > 2%: report and analyse
  which concept families have low activation variance (indicator that
  the dataset's discriminative features are not well covered by the
  current vocabulary)
- Datasets where bypass ratio > 0.3: flag as requiring adversarial training
  or concept vocabulary extension; do not cherry-pick results

---

## Implementation tasks

### TC-01 · TCRPClassifier module

**File:** `model/classifier.py`

Subclass or wrap `TCRPForecaster` with:

- Decoder: `nn.Linear(K, C)` with softmax output
- Loss: `F.cross_entropy`
- Relevance initialisation: `R_out = y_hat[k_star]` (predicted class logit)
- All other components identical to forecaster

Note: the TCRP analysis pass (T-17) applies unchanged. The
concept-conditional temporal maps now answer "which historical windows
expressed which trend concept to push the model toward class $k^*$?"

### TC-02 · Classification data loaders

**File:** `data/classification_datasets.py`

Loaders for: ECG5000, MIT-BIH, Sleep-EDF, CWRU, UCI-HAR,
EthanolConcentration, UCR archive (generic loader).

Each loader returns `(x, y)` pairs:

- `x: Tensor (T,)` for univariate, `(T, V)` for multivariate
- `y: int` class label

Preprocessing:

- Z-score normalise per channel using train-set statistics only
- For imbalanced datasets (MIT-BIH, Sleep-EDF): report class distribution;
  use weighted cross-entropy loss with class frequency inverse weights
- Patient-level splits for clinical datasets (EXP-C01, C02, C03):
  no patient in both train and test

### TC-03 · Classification metrics

**File:** `eval/classification_metrics.py`

- `accuracy(y_pred, y_true) -> float`
- `macro_f1(y_pred, y_true) -> float`
- `per_class_accuracy(y_pred, y_true) -> dict`
- `confusion_matrix(y_pred, y_true) -> Tensor`
- `evaluate_all(model, loader) -> dict`

### TC-04 · Concept relevance profile per class

**File:** `eval/concept_class_profile.py`

For each class $c$, compute the mean concept relevance vector over all
test samples predicted as class $c$:

```python
def class_concept_profiles(
    model: TCRPClassifier,
    analyser: TCRPAnalyser,
    loader: DataLoader,
    concept_names: list[str],
) -> dict:
    """
    Returns {class_idx: {concept_name: mean_relevance}} for all classes.
    Positive mean_relevance: concept pushes toward this class.
    Negative: concept pushes away from this class.
    Used to characterise what each class "looks like" in concept space.
    """
```

### TC-05 · Experiment runner scripts

```
python scripts/run_classification.py \
    --experiment EXP-C01 \
    --seed 42 \
    --adversarial False
```

Runs training, evaluation, all diagnostics (T-29–T-35), and saves:

- `results/EXP-C01/metrics.json` — accuracy, F1, CAS, bypass ratio
- `results/EXP-C01/concept_profiles.json` — per-class concept relevance
- `results/EXP-C01/overfit_report.json` — T-34 dashboard output
- `results/EXP-C01/purity_log.csv` — T\*-05 concept purity over epochs

```
python scripts/run_ucr_benchmark.py \
    --datasets all_15 \
    --seeds 0 1 2 3 4
```

Runs EXP-C09 across all 15 datasets and 5 seeds; saves aggregate table.

### TC-06 · Qualitative visualisation per experiment

For each of EXP-C01, C04, C07:

- Plot 1: concept relevance bar chart for 3 representative test samples
  per class (correct predictions only)
- Plot 2: concept-conditional temporal maps overlaid on raw series,
  one colour per concept family, stacked area style
- Plot 3: class concept profiles as a heatmap (classes × concepts)
- Plot 4 (EXP-C01 only): ECG waveform with TCRP temporal map overlaid,
  compared side-by-side with gradient saliency on same sample

---

## Timeline suggestion

| Phase             | Tasks               | Prerequisite                          |
| ----------------- | ------------------- | ------------------------------------- |
| Setup             | TC-01, TC-02, TC-03 | T-13 (base model) complete            |
| Core clinical     | EXP-C01, C02        | TC-01–TC-03                           |
| Expert annotation | EXP-C01 CAS         | EXP-C01 training done                 |
| Industrial        | EXP-C04, C05        | TC-01–TC-03                           |
| Financial         | EXP-C07, C08        | TC-01–TC-03                           |
| UCR breadth       | EXP-C09             | TC-01–TC-03                           |
| Ablations         | EXP-C10–C13         | EXP-C01, C04 done                     |
| Adversarial       | EXP-C11             | T\*-04 (adversarial trainer) complete |
| Visualisation     | TC-06               | All experiments done                  |
