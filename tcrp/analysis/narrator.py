"""Domain-specific LLM interpretation of TCRP explanations.

Usage::

    from tcrp.analysis.narrator import TCRPNarrator

    narrator = TCRPNarrator(dataset="FX")
    narrative = narrator.narrate(
        explanation=expl,          # TCRPExplanation
        concept_names=names,       # list[str] from model.scorer.concept_names
        sample_idx=0,              # which batch item to describe
        top_k=6,                   # concepts to surface
    )
    print(narrative)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic
import numpy as np

from tcrp.analysis.tcrp_analysis import TCRPExplanation

# ── Domain registry ────────────────────────────────────────────────────────


@dataclass
class _DomainSpec:
    name: str  # human-readable domain label
    task: str  # what the model is classifying/predicting
    class_labels: dict[int, str]  # class index → plain-English label
    unit: str  # what a segment represents, e.g. "days"
    system_persona: str  # one-sentence expert persona for the prompt


_DOMAINS: dict[str, _DomainSpec] = {
    "FX": _DomainSpec(
        name="Foreign Exchange",
        task="FX regime classification (trending / mean-reverting / random walk)",
        class_labels={0: "trending", 1: "mean-reverting", 2: "random walk"},
        unit="trading days",
        system_persona=(
            "You are a quantitative analyst specialising in FX market microstructure "
            "and regime detection."
        ),
    ),
    "SP500_A": _DomainSpec(
        name="S&P 500 (binary)",
        task="S&P 500 recession vs expansion regime classification",
        class_labels={0: "expansion", 1: "recession"},
        unit="trading days",
        system_persona=(
            "You are a macroeconomic analyst specialising in equity market regime detection."
        ),
    ),
    "SP500_B": _DomainSpec(
        name="S&P 500 (VIX regime)",
        task="S&P 500 volatility regime classification (low / medium / high)",
        class_labels={
            0: "low-volatility",
            1: "medium-volatility",
            2: "high-volatility",
        },
        unit="trading days",
        system_persona=(
            "You are a risk analyst specialising in equity volatility regimes."
        ),
    ),
    "ECG5000": _DomainSpec(
        name="ECG (arrhythmia)",
        task="cardiac arrhythmia classification from ECG beats",
        class_labels={
            0: "normal sinus rhythm",
            1: "R-on-T PVC",
            2: "PVC",
            3: "paced beat",
            4: "other/unknown",
        },
        unit="ECG samples (~200 Hz)",
        system_persona=(
            "You are a clinical cardiologist interpreting ECG beat morphology."
        ),
    ),
    "MITBIH": _DomainSpec(
        name="MIT-BIH (arrhythmia)",
        task="ECG beat classification (AAMI standard: N/S/V/F/Q)",
        class_labels={
            0: "normal/bundle-branch-block",
            1: "supraventricular ectopic",
            2: "ventricular ectopic",
            3: "fusion beat",
            4: "unknown/paced",
        },
        unit="ECG samples",
        system_persona=(
            "You are a clinical electrophysiologist interpreting ECG beat morphology "
            "using the AAMI standard."
        ),
    ),
    "SleepEDF": _DomainSpec(
        name="Sleep-EDF (sleep staging)",
        task="sleep stage classification from EEG epochs",
        class_labels={
            0: "Wake",
            1: "Stage N1 (light sleep)",
            2: "Stage N2 (intermediate sleep)",
            3: "Stage N3 (deep sleep)",
            4: "REM sleep",
        },
        unit="EEG samples (~100 Hz)",
        system_persona=(
            "You are a sleep medicine specialist interpreting polysomnography EEG epochs."
        ),
    ),
    "CWRU": _DomainSpec(
        name="CWRU Bearing",
        task="bearing fault detection and severity classification",
        class_labels={
            0: "normal (healthy bearing)",
            1: "inner race fault",
            2: "outer race fault",
            3: "ball fault",
        },
        unit="vibration samples",
        system_persona=(
            "You are a mechanical engineer specialising in rotating machinery "
            "condition monitoring and fault diagnosis."
        ),
    ),
    "UCIHAR": _DomainSpec(
        name="UCI-HAR (human activity)",
        task="human activity recognition from inertial sensor data",
        class_labels={
            0: "walking",
            1: "walking upstairs",
            2: "walking downstairs",
            3: "sitting",
            4: "standing",
            5: "lying",
        },
        unit="accelerometer/gyroscope samples",
        system_persona=(
            "You are a biomedical engineer analysing inertial sensor data for "
            "human activity recognition."
        ),
    ),
    "Ethanol": _DomainSpec(
        name="Ethanol Concentration",
        task="ethanol concentration level classification",
        class_labels={0: "35%", 1: "38%", 2: "40%", 3: "45%"},
        unit="sensor readings",
        system_persona=(
            "You are an analytical chemist interpreting chemical sensor time series "
            "for beverage quality control."
        ),
    ),
    # ── Forecasting datasets ──────────────────────────────────────────────────
    "ETTh1": _DomainSpec(
        name="ETT-h1 (Electricity Transformer Temperature)",
        task="oil temperature forecasting from power-load and transformer features",
        class_labels={},
        unit="hours",
        system_persona=(
            "You are a power-systems engineer analysing electricity transformer "
            "temperature dynamics and load patterns."
        ),
    ),
    "ETTm2": _DomainSpec(
        name="ETT-m2 (Electricity Transformer Temperature)",
        task="oil temperature forecasting from power-load and transformer features",
        class_labels={},
        unit="15-minute intervals",
        system_persona=(
            "You are a power-systems engineer analysing electricity transformer "
            "temperature dynamics and load patterns."
        ),
    ),
    "Weather": _DomainSpec(
        name="Weather (Wet-Bulb Temperature)",
        task="wet-bulb temperature forecasting from meteorological features",
        class_labels={},
        unit="10-minute intervals",
        system_persona=(
            "You are a meteorologist interpreting atmospheric time-series patterns "
            "for short-range temperature forecasting."
        ),
    ),
    "ExchangeRate": _DomainSpec(
        name="Exchange Rate",
        task="foreign exchange rate forecasting",
        class_labels={},
        unit="trading days",
        system_persona=(
            "You are a quantitative analyst specialising in FX time-series modelling "
            "and short-horizon rate forecasting."
        ),
    ),
    "GEFCOM2014": _DomainSpec(
        name="GEFCom2014 (Energy)",
        task="probabilistic electricity load forecasting",
        class_labels={},
        unit="hours",
        system_persona=(
            "You are an energy analyst specialising in electricity demand forecasting "
            "and grid load modelling."
        ),
    ),
}

_FALLBACK_DOMAIN = _DomainSpec(
    name="time series",
    task="time-series classification",
    class_labels={},
    unit="time steps",
    system_persona="You are a machine-learning scientist interpreting time-series model explanations.",
)


# ── Narrator ───────────────────────────────────────────────────────────────


class TCRPNarrator:
    """Generate a natural-language interpretation of a TCRP explanation.

    Args:
        dataset:     Dataset name (e.g. ``"FX"``, ``"ECG5000"``). Used to
                     select the domain persona and class labels.
        model:       Anthropic model ID. Defaults to ``claude-opus-4-8``.
        max_tokens:  Maximum tokens for the narrative response.
    """

    def __init__(
        self,
        dataset: str = "FX",
        model: str = "claude-opus-4-8",
        max_tokens: int = 512,
    ) -> None:
        """Narrator constructor.

        Args:
            dataset (str, optional): _description_. Defaults to "FX".
            model (str, optional): _description_. Defaults to "claude-opus-4-8".
            max_tokens (int, optional): _description_. Defaults to 512.
        """
        self.domain = _DOMAINS.get(dataset, _FALLBACK_DOMAIN)
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def narrate(
        self,
        explanation: TCRPExplanation,
        concept_names: list[str],
        sample_idx: int = 0,
        top_k: int = 6,
        h_star: int | None = None,
    ) -> str:
        """Return a 2–4 sentence domain-specific narrative for one sample.

        Args:
            explanation:   ``TCRPExplanation`` from ``TCRPAnalyser.analyse()``.
            concept_names: Ordered list of concept names (``model.scorer.concept_names``).
            sample_idx:    Batch dimension index of the sample to describe.
            top_k:         How many top concepts to surface in the narrative.
            h_star:        Forecast horizon step being explained (forecasting only).
                           Pass ``None`` for classification tasks.

        Returns:
            Plain-text narrative string.
        """
        payload = self._build_payload(
            explanation, concept_names, sample_idx, top_k, h_star
        )
        return self._call_llm(payload)

    def narrate_batch(
        self,
        explanation: TCRPExplanation,
        concept_names: list[str],
        top_k: int = 6,
        h_star: int | None = None,
    ) -> list[str]:
        """Return narratives for every sample in the explanation batch."""
        B = explanation.R_h.shape[0]
        return [
            self.narrate(explanation, concept_names, i, top_k, h_star) for i in range(B)
        ]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_payload(
        self,
        expl: TCRPExplanation,
        concept_names: list[str],
        idx: int,
        top_k: int,
        h_star: int | None = None,
    ) -> dict:
        """Extract numeric scores and build the structured prompt payload."""
        # --- concept relevances (R_h: concept-level, shape (B, K)) ---
        r_h = expl.R_h[idx].detach().cpu().numpy()  # (K,)
        top_idx = np.argsort(np.abs(r_h))[::-1][:top_k]
        top_concepts = [
            {
                "concept": (
                    concept_names[int(i)]
                    if int(i) < len(concept_names)
                    else f"concept_{i}"
                ),
                "relevance": float(r_h[i]),
            }
            for i in top_idx
        ]

        # --- predicted class ---
        if expl.k_stars is not None:
            pred_class_idx = int(expl.k_stars[idx].item())
            pred_class_label = self.domain.class_labels.get(
                pred_class_idx, f"class {pred_class_idx}"
            )
        else:
            pred_class_idx = None
            pred_class_label = "unknown"

        # --- segment-level top concept per segment (R_A: (B,N,K)) ---
        n_segs = expl.R_A.shape[1]
        r_a = expl.R_A[idx].detach().cpu().numpy()  # (N, K)
        seg_dominant = []
        for n in range(min(n_segs, 8)):  # cap at 8 segs for brevity
            dom_k = int(np.argmax(np.abs(r_a[n])))
            seg_dominant.append(
                concept_names[dom_k]
                if dom_k < len(concept_names)
                else f"concept_{dom_k}"
            )

        return {
            "domain": self.domain,
            "pred_class_idx": pred_class_idx,
            "pred_class_label": pred_class_label,
            "h_star": h_star,
            "top_concepts": top_concepts,
            "n_segments": n_segs,
            "seg_dominant": seg_dominant,
            "segment_length": expl.L,
        }

    def _build_prompt(self, payload: dict) -> str:
        domain: _DomainSpec = payload["domain"]
        h_star: int | None = payload["h_star"]
        top_k_lines = "\n".join(
            f"  {i+1}. {c['concept']:30s}  relevance = {c['relevance']:+.4f}"
            for i, c in enumerate(payload["top_concepts"])
        )
        seg_lines = ", ".join(
            f"seg{i+1}={name}" for i, name in enumerate(payload["seg_dominant"])
        )

        header = (
            f"## TCRP Explanation Summary\n\n"
            f"**Domain:** {domain.name}\n"
            f"**Task:** {domain.task}\n"
        )
        ts_line = (
            f"**Time series:** {payload['n_segments']} segments × "
            f"{payload['segment_length']} {domain.unit}\n\n"
        )
        concepts_block = (
            f"### Top-{len(payload['top_concepts'])} concept relevances\n"
            f"(Positive = drives prediction; negative = suppresses it)\n\n"
            f"{top_k_lines}\n\n"
            f"### Dominant concept per segment\n"
            f"{seg_lines}\n\n"
            f"---\n\n"
        )

        if h_star is not None:
            target_line = f"**Forecast horizon:** h*={h_star +1} steps ahead\n"
            instruction = (
                f"Write a 2–4 sentence domain interpretation of which temporal patterns "
                f"the model relies on to forecast **{h_star+1} steps ahead** in this "
                f"{domain.name} series. "
                f"Reference the specific concepts by name. "
                f"Be precise and concise; do not pad with generic disclaimers."
            )
        else:
            pred_label = payload["pred_class_label"]
            target_line = f"**Predicted class:** {pred_label}\n"
            instruction = (
                f"Write a 2–4 sentence domain interpretation of why the model "
                f"predicted **{pred_label}** for this sample. "
                f"Reference the specific concepts by name. "
                f"Be precise and concise; do not pad with generic disclaimers."
            )

        return header + target_line + ts_line + concepts_block + instruction

    def _call_llm(self, payload: dict) -> str:
        domain: _DomainSpec = payload["domain"]
        system = (
            f"{domain.system_persona} "
            f"You explain machine-learning model decisions in plain, precise language "
            f"that a domain expert would find immediately actionable."
        )
        user_prompt = self._build_prompt(payload)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return next(
            (block.text for block in response.content if block.type == "text"), ""
        )
