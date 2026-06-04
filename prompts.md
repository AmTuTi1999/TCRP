# TCRP Training Prompts

All commands run from the project root.

The training pipeline uses **Hydra** for config composition.
`configs/train.yaml` is the entry point; it composes `models/tcrp`, a dataset,
and a trainer via the `defaults:` list.  Override any field on the command line.

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
python3 -m tcrp.pipelines.train

# Change dataset or horizon inline
python3 -m tcrp.pipelines.train datasets=ettm2 H=192
python3 -m tcrp.pipelines.train datasets=weather H=336

# Use a pre-built experiment override file
python3 -m tcrp.pipelines.train +experiment=etth1_H96
python3 -m tcrp.pipelines.train +experiment=etth1_H192
python3 -m tcrp.pipelines.train +experiment=etth1_H336
python3 -m tcrp.pipelines.train +experiment=etth1_H720

python3 -m tcrp.pipelines.train +experiment=ettm2_H96
python3 -m tcrp.pipelines.train +experiment=ettm2_H192
python3 -m tcrp.pipelines.train +experiment=ettm2_H336
python3 -m tcrp.pipelines.train +experiment=ettm2_H720

python3 -m tcrp.pipelines.train +experiment=weather_H96
python3 -m tcrp.pipelines.train +experiment=weather_H192
python3 -m tcrp.pipelines.train +experiment=weather_H336
python3 -m tcrp.pipelines.train +experiment=weather_H720

python3 -m tcrp.pipelines.train +experiment=exchange_rate_H96
python3 -m tcrp.pipelines.train +experiment=exchange_rate_H192
python3 -m tcrp.pipelines.train +experiment=exchange_rate_H336
python3 -m tcrp.pipelines.train +experiment=exchange_rate_H720
```

---

## TCRP — Adversarial

```bash
python3 -m tcrp.pipelines.train +experiment=etth1_H96 models.adversarial=true

python3 -m tcrp.pipelines.train +experiment=etth1_H96 \
    models.adversarial=true \
    models.alpha_max=1.0 \
    models.warmup_epochs=20 \
    models.lambda3=0.01
```

---

## Baselines

```bash
# NBeats
python3 -m tcrp.pipelines.train +experiment=etth1_H96 \
    model_type=nbeats baseline_hidden=256 baseline_layers=3

# LSTM
python3 -m tcrp.pipelines.train +experiment=etth1_H96 \
    model_type=lstm baseline_hidden=128 baseline_layers=2

# TCN
python3 -m tcrp.pipelines.train +experiment=etth1_H96 \
    model_type=tcn baseline_hidden=64 baseline_layers=4
```

---

## Adversarial vs Standard Comparison

```bash
python3 scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42

python3 scripts/adversarial_compare.py --dataset ETTh1 --H 96 --seed 42 \
    --out results/adv_compare_etth1_H96.json
```

---

## Evaluate a Saved Checkpoint

```bash
python3 -m tcrp.pipelines.evaluate \
    --config configs/train.yaml \
    --checkpoint checkpoints/etth1_H96_best.pt

python3 -m tcrp.pipelines.evaluate \
    --config configs/train.yaml \
    --checkpoint checkpoints/etth1_H96_best.pt \
    --out results/etth1_H96_test.json
```

---

## Common Overrides

| Field | Scope | Example |
|---|---|---|
| `H` | horizon | `H=192` |
| `T` | look-back window | `T=720` |
| `datasets` | swap dataset | `datasets=ettm2` |
| `models.encoder_hidden` | latent dim | `models.encoder_hidden=128` |
| `models.tcn_encoder_n_layers` | TCN depth | `models.tcn_encoder_n_layers=6` |
| `models.attention_hidden` | pool MLP dim | `models.attention_hidden=64` |
| `models.lambda1` | alignment weight | `models.lambda1=0.05` |
| `trainers.lr` | learning rate | `trainers.lr=5e-4` |
| `trainers.max_epochs` | training epochs | `trainers.max_epochs=200` |
| `trainers.batch_size` | batch size | `trainers.batch_size=64` |
| `seed` | random seed | `seed=0` |
| `run_name` | checkpoint prefix | `run_name=etth1_h96_run1` |

---

## Config Locations

| Type | Path |
|---|---|
| Entry point | `configs/train.yaml` |
| Model defaults | `configs/models/tcrp.yaml` |
| Dataset configs | `configs/datasets/{dataset}.yaml` |
| Trainer defaults | `configs/trainers/tcrp_trainer.yaml` |
| Experiment overrides | `configs/experiments/{dataset}_H{H}.yaml` |
