"""Fase 2 — Backtesting honesto.

Protocolo: entrenar SOLO con partidos anteriores a una fecha de corte, predecir los
partidos de un torneo/período posterior, y medir calidad de las probabilidades.

Comparamos dos modelos:
  - `baseline_rates`: predice siempre las tasas base de H/D/A del set de train.
    Es el "no-modelo": si el Elo no le gana a esto, no aprendiste nada.
  - `elo`: el baseline Elo de baseline_elo.py.

Métricas (un modelo de probabilidades NO se juzga por accuracy seca):
  - log-loss  → penaliza fuerte estar seguro y equivocado. Más bajo = mejor.
  - Brier     → error cuadrático sobre las probabilidades. Más bajo = mejor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_elo import OUTCOME_ORDER, OutcomeModel, compute_elo_history
from fetch_data import load_matches


def _onehot(outcomes: np.ndarray) -> np.ndarray:
    """outcomes (['H','D','A',...]) → matriz one-hot en orden [A, D, H]."""
    idx = {c: i for i, c in enumerate(OUTCOME_ORDER)}
    oh = np.zeros((len(outcomes), 3))
    for i, o in enumerate(outcomes):
        oh[i, idx[o]] = 1.0
    return oh


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    oh = _onehot(outcomes)
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(oh * np.log(p)) / len(outcomes))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    oh = _onehot(outcomes)
    return float(np.sum((probs - oh) ** 2) / len(outcomes))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Accuracy del argmax — solo informativo, NO es la métrica principal."""
    pred = np.array([OUTCOME_ORDER[i] for i in probs.argmax(axis=1)])
    return float((pred == outcomes).mean())


def backtest(cutoff: str = "2022-01-01", tournament_filter: str | None = "FIFA World Cup") -> None:
    matches = load_matches()

    # Elo se calcula sobre TODA la historia en orden cronológico (cada partido usa
    # solo el Elo pre-partido, así que no hay leakage temporal aunque calculemos todo).
    with_elo, _ = compute_elo_history(matches)

    cutoff_ts = pd.Timestamp(cutoff)
    train = with_elo[with_elo["date"] < cutoff_ts]
    test = with_elo[with_elo["date"] >= cutoff_ts]
    if tournament_filter:
        # Match exacto: "FIFA World Cup" NO incluye "FIFA World Cup qualification".
        test = test[test["tournament"] == tournament_filter]

    if len(test) == 0:
        print(f"[!] No hay partidos de test (cutoff={cutoff}, filtro={tournament_filter!r}).")
        return

    # --- Modelo Elo: el mapeo outcome se entrena SOLO con train ---
    elo_model = OutcomeModel.fit(train)
    elo_probs = elo_model.predict_proba(test["elo_diff"].to_numpy())

    # --- Baseline trivial: tasas base de train ---
    base_rates = (
        train["outcome"].value_counts(normalize=True).reindex(OUTCOME_ORDER).fillna(0).to_numpy()
    )
    base_probs = np.tile(base_rates, (len(test), 1))

    y_test = test["outcome"].to_numpy()

    print(f"\nCutoff: {cutoff}  |  filtro torneo: {tournament_filter!r}")
    print(f"Train: {len(train):,} partidos  |  Test: {len(test):,} partidos\n")
    print(f"{'modelo':<16}{'log-loss':>12}{'brier':>10}{'accuracy':>10}")
    print("-" * 48)
    for name, probs in [("tasas base", base_probs), ("elo", elo_probs)]:
        print(f"{name:<16}{log_loss(probs, y_test):>12.4f}"
              f"{brier(probs, y_test):>10.4f}{accuracy(probs, y_test):>10.1%}")

    ll_base = log_loss(base_probs, y_test)
    ll_elo = log_loss(elo_probs, y_test)
    delta = (ll_base - ll_elo) / ll_base
    verdict = "[OK] Elo le gana al no-modelo" if ll_elo < ll_base else "[X] Elo NO mejora - revisar"
    print(f"\n{verdict}  (log-loss {delta:+.1%} vs tasas base)")


if __name__ == "__main__":
    # Por defecto: entrenar con todo hasta 2022, evaluar el Mundial de Qatar 2022.
    backtest(cutoff="2022-01-01", tournament_filter="FIFA World Cup")
