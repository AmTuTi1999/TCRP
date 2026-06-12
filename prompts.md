# TCRP Training Prompts

All commands run from the project root.

The training pipeline uses **Hydra** for config composition.
`configs/train.yaml` is the entry point; it composes `models/tcrp`, a dataset,
and a trainer via the `defaults:` list. Override any field on the command line.

```
configs/
  train.yaml            ← entry point (defaults: models/tcrp + datasets/etth1 + trainers/tcrp_trainer)
  models/tcrp.yaml      ← model architecture defaults (encoder, attention, loss weights)
  datasets/             ← pure dataset configs (etth1, ettm2, weather, exchange_rate, gefcom2014)
  trainers/             ← trainer defaults (lr, batch_size, patience, …)
  experiments/          ← per-horizon experiment overrides (etth1_H96, ettm2_H192, …)
```

---

## TCRP — Standard

```bash
# Default (ETTh1, H=96)
poetry run python train_script.py

# Change dataset or horizon inline
poetry run python train_script.py datasets=ettm2 H=192
poetry run python train_script.py datasets=weather H=336
```

---

## TCRP — Adversarial

```bash
poetry run python train_script.py +experiment=etth1_tcrp models.adversarial=true

poetry run python train_script.py +experiment=etth1_tcrp \
    models.adversarial=true \
    models.alpha_max=1.0 \
    models.warmup_epochs=20 \
    models.lambda3=0.01
```

---

## Baselines

```bash
# NBeats
poetry run python train_script.py +experiment=etth1_baseline \
     ++model_type=nbeats +models.baseline_hidden=256 +models.baseline_layers=3

# LSTM
poetry run python train_script.py +experiment=etth1_baseline \
     ++model_type=lstm +models.baseline_hidden=128 +models.baseline_layers=2

# TCN
poetry run python train_script.py +experiment=etth1_baseline \
    ++model_type=tcn +models.baseline_hidden=64 +models.baseline_layers=4
```

---

## Adversarial vs Standard Comparison

```bash
poetry run python scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42

poetry run python scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42 \
    --out results/adv_compare_etth1_tcrp.json
```

---

## Evaluate a Saved Checkpoint

```bash
poetry run python -m tcrp.pipelines.evaluate \
    --config configs/train.yaml \
    --checkpoint checkpoints/etth1_tcrp_best.pt

poetry run python -m tcrp.pipelines.evaluate \
    --config configs/train.yaml \
    --checkpoint checkpoints/etth1_tcrp_best.pt \
    --out results/etth1_tcrp_test.json
```

---

## Common Overrides

| Field                         | Scope             | Example                         |
| ----------------------------- | ----------------- | ------------------------------- |
| `H`                           | horizon           | `H=192`                         |
| `T`                           | look-back window  | `T=720`                         |
| `datasets`                    | swap dataset      | `datasets=ettm2`                |
| `models.encoder_hidden`       | latent dim        | `models.encoder_hidden=128`     |
| `models.tcn_encoder_n_layers` | TCN depth         | `models.tcn_encoder_n_layers=6` |
| `models.attention_hidden`     | pool MLP dim      | `models.attention_hidden=64`    |
| `models.lambda1`              | alignment weight  | `models.lambda1=0.05`           |
| `trainers.lr`                 | learning rate     | `trainers.lr=5e-4`              |
| `trainers.max_epochs`         | training epochs   | `trainers.max_epochs=200`       |
| `trainers.batch_size`         | batch size        | `trainers.batch_size=64`        |
| `seed`                        | random seed       | `seed=0`                        |
| `run_name`                    | checkpoint prefix | `run_name=etth1_h96_run1`       |

---

## Config Locations

| Type                 | Path                                         |
| -------------------- | -------------------------------------------- |
| Entry point          | `configs/train.yaml`                         |
| Model defaults       | `configs/models/tcrp.yaml`                   |
| Dataset configs      | `configs/datasets/{dataset}.yaml`            |
| Trainer defaults     | `configs/trainers/tcrp_trainer.yaml`         |
| Experiment overrides | `configs/experiments/{dataset}_{model}.yaml` |

---

## Classification Experiments (EXP-C01 – EXP-C08)

Classification entry point is `scripts/run_classification.py`.
Config tree mirrors the forecasting tree under `train_classification.yaml`.

```
configs/
  train_classification.yaml           ← entry point (defaults: tcrp_classifier + dataset + classification_trainer)
  models/tcrp_classifier.yaml         ← classifier model defaults
  datasets/{ecg5000,mitbih,…}.yaml    ← classification dataset configs
  trainers/classification_trainer.yaml
  experiments/classification/         ← per-experiment overrides (exp_c01_ecg5000, …)
```

### Set 1 — Physiological / Clinical

```bash
# EXP-C01 · ECG5000 arrhythmia (T=140, C=5) — data available locally
poetry run python scripts/run_classification.py --experiment EXP-C01 --seed 42

# Multiple seeds
for seed in 0 1 2 3 4; do
    poetry run python scripts/run_classification.py --experiment EXP-C01 --seed $seed
done

# EXP-C02 · MIT-BIH heartbeat (T=187, C=5) — data available locally
poetry run python scripts/run_classification.py --experiment EXP-C02 --seed 42

# EXP-C03 · Sleep-EDF sleep stages (T=3000, C=5) — requires download
poetry run python scripts/run_classification.py --experiment EXP-C03 --seed 42
```

### Set 2 — Industrial / Fault Detection

```bash
# EXP-C04 · CWRU bearing faults (T=1024, C=4) — requires download
poetry run python scripts/run_classification.py --experiment EXP-C04 --seed 42

# EXP-C05 · UCI-HAR human activity (T=128, C=6) — requires download
poetry run python scripts/run_classification.py --experiment EXP-C05 --seed 42

# EXP-C06 · EthanolConcentration (T=1751, C=4, no periodicity) — requires download
poetry run python scripts/run_classification.py --experiment EXP-C06 --seed 42
```

### Set 3 — Financial / Economic

```bash
# EXP-C07 Task A · S&P 500 recession classification (C=2) — requires download
poetry run python scripts/run_classification.py --experiment EXP-C07-A --seed 42

# EXP-C07 Task B · S&P 500 volatility regime (C=3) — requires download
poetry run python scripts/run_classification.py --experiment EXP-C07-B --seed 42

# EXP-C08 · FX EURUSD trend/mean-reversion regime (T=21, C=3) — data available locally
poetry run python scripts/run_classification.py --experiment EXP-C08 --seed 42
```

#### Analyse

````bash
poetry run python crp_analysis_script.py datasets=ecg5000 ++checkpoint_dir="checkpoints/EXP-C01_seed42_best.pt" ++run_name="ECG_5000_classification" ++model_type="tcrp_classifier"
### Adversarial training variant

```bash
poetry run python scripts/run_classification.py --experiment EXP-C01 --seed 42 --adversarial
poetry run python scripts/run_classification.py --experiment EXP-C07-A --seed 42 --adversarial
````

### Classification overrides table

| Field              | Example                         |
| ------------------ | ------------------------------- |
| `--experiment`     | `EXP-C01` … `EXP-C08`           |
| `--seed`           | `--seed 0`                      |
| `--adversarial`    | enables GRL training            |
| `--results_dir`    | `--results_dir results/`        |
| `--checkpoint_dir` | `--checkpoint_dir checkpoints/` |

### Classification config locations

| Type                 | Path                                                    |
| -------------------- | ------------------------------------------------------- |
| Entry point          | `configs/train_classification.yaml`                     |
| Model defaults       | `configs/models/tcrp_classifier.yaml`                   |
| Dataset configs      | `configs/datasets/{ecg5000,mitbih,cwru,…}.yaml`         |
| Trainer defaults     | `configs/trainers/classification_trainer.yaml`          |
| Experiment overrides | `configs/experiments/classification/exp_c0{1-8}_*.yaml` |

---

## Baseline Classification Experiments

Entry point is `scripts/run_baseline_classification.py`.
Models: `mlp` · `lstm` · `fcn` · `resnet` · `nbeats`
Output format is identical to `run_classification.py` for direct comparison.

```bash
# Single model + experiment
poetry run python scripts/run_baseline_classification.py --experiment EXP-C01 --model fcn

# All models for one experiment
poetry run python scripts/run_baseline_classification.py --experiment EXP-C01 --model all

# One model across all experiments
poetry run python scripts/run_baseline_classification.py --experiment all --model resnet

# Full sweep (all models × all experiments)
poetry run python scripts/run_baseline_classification.py --experiment all --model all

# With custom seed / output dirs
poetry run python scripts/run_baseline_classification.py \
    --experiment EXP-C02 --model fcn --seed 0 \
    --results_dir results/ --checkpoint_dir checkpoints/
```

### Baseline model descriptions

| Model    | Architecture                                            | Reference            |
| -------- | ------------------------------------------------------- | -------------------- |
| `mlp`    | Flatten → 3×(Linear 500→ReLU→Dropout)                   | Wang et al. 2017     |
| `lstm`   | Bidirectional LSTM (2 layers, hidden=128) → linear head | —                    |
| `fcn`    | Conv(128,k=8)→Conv(256,k=5)→Conv(128,k=3) + GAP         | Wang et al. 2017     |
| `resnet` | 3 residual blocks (1→64→128→128 ch) + GAP               | Wang et al. 2017     |
| `nbeats` | N-BEATS stack (backcast residuals, accumulated logits)  | Oreshkin et al. 2020 |

### Baseline overrides table

| Flag               | Example                            |
| ------------------ | ---------------------------------- |
| `--experiment`     | `EXP-C01` … `EXP-C08` or `all`     |
| `--model`          | `fcn`, `resnet`, `mlp`, … or `all` |
| `--seed`           | `--seed 0`                         |
| `--results_dir`    | `--results_dir results/`           |
| `--checkpoint_dir` | `--checkpoint_dir checkpoints/`    |

Results are saved to `results/BL-{MODEL}_{EXPERIMENT}_seed{N}/metrics.json`.
