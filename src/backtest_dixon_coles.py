"""Backtest honesto: Dixon-Coles vs baseline Elo en Mundiales.

Conclusión (ver README): con validación out-of-sample en torneos separados, Dixon-Coles
NO le gana consistentemente al baseline Elo. Pierde en 2014 y 2018, solo gana en 2022;
ningún hiperparámetro fijo supera al Elo en los tres. Por eso el modelo de producción
sigue siendo el Elo. Este script deja la evidencia reproducible.

Correr: python src/backtest_dixon_coles.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_elo import OUTCOME_ORDER, OutcomeModel, compute_elo_history
from dixon_coles import DixonColes
from fetch_data import load_matches

WORLD_CUPS = [
    ("2014", "2014-06-01", "2014-07-20"),
    ("2018", "2018-06-01", "2018-07-20"),
    ("2022", "2022-11-01", "2022-12-25"),
]
ALPHA, HALF_LIFE = 7e-4, 365 * 7  # parámetros DC fijos a priori (no tuneados por test)


def _onehot(o):
    idx = {c: i for i, c in enumerate(OUTCOME_ORDER)}
    M = np.zeros((len(o), 3))
    for i, x in enumerate(o):
        M[i, idx[x]] = 1
    return M


def log_loss(p, o, eps=1e-12):
    return float(-np.sum(_onehot(o) * np.log(np.clip(p, eps, 1))) / len(o))


def accuracy(p, o):
    pred = np.array([OUTCOME_ORDER[i] for i in p.argmax(1)])
    return float((pred == o).mean())


def main() -> None:
    m = load_matches()
    we, _ = compute_elo_history(m)
    print(f"DC fijo: alpha={ALPHA}, half-life={HALF_LIFE // 365}y\n")
    print(f"{'Mundial':<9}{'n':>4}{'elo_ll':>9}{'dc_ll':>9}{'mejora':>9}"
          f"{'elo_acc':>9}{'dc_acc':>8}")
    print("-" * 57)
    elo_wins = dc_wins = 0
    for year, start, end in WORLD_CUPS:
        cutoff = pd.Timestamp(start)
        test = we[(we["date"] >= start) & (we["date"] <= end)
                  & (we["tournament"] == "FIFA World Cup")]
        if len(test) == 0:
            continue
        y = test["outcome"].to_numpy()
        em = OutcomeModel.fit(we[we["date"] < cutoff])
        elo_p = em.predict_proba(test["elo_diff"].to_numpy())
        dc = DixonColes.fit(m, ref_date=cutoff, alpha=ALPHA, half_life_days=HALF_LIFE)
        dp = np.array([
            [dc.predict(r.home_team, r.away_team,
                        home_for=(r.home_team if not r.neutral else None))[k]
             for k in ("away_win", "draw", "home_win")]
            for r in test.itertuples()])
        ell, dll = log_loss(elo_p, y), log_loss(dp, y)
        imp = (ell - dll) / ell
        if dll < ell:
            dc_wins += 1
        else:
            elo_wins += 1
        print(f"{year:<9}{len(test):>4}{ell:>9.4f}{dll:>9.4f}{imp:>8.1%}"
              f"{accuracy(elo_p, y):>9.1%}{accuracy(dp, y):>8.1%}")
    print(f"\nVeredicto: Elo gana {elo_wins} / DC gana {dc_wins} de {len(WORLD_CUPS)} "
          f"Mundiales -> el modelo de produccion sigue siendo Elo.")


if __name__ == "__main__":
    main()
