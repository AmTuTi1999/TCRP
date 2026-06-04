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
python3 scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42

python3 scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42 \
    --out results/adv_compare_etth1_tcrp.json
```

---

## Evaluate a Saved Checkpoint

```bash
python3 -m tcrp.pipelines.evaluate \
    --config configs/train.yaml \
    --checkpoint checkpoints/etth1_tcrp_best.pt

python3 -m tcrp.pipelines.evaluate \
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
