"""Fase 2 — Baseline Elo.

Dos piezas:

1. `compute_elo_history`: recorre la historia de partidos y asigna a cada selección
   un rating Elo que se actualiza partido a partido. Para cada partido guarda el Elo
   PRE-partido de ambos equipos (clave: nunca usar info del futuro).

2. `OutcomeModel`: mapea la diferencia de Elo (con ventaja de localía) a probabilidades
   1/X/2 con una regresión logística multinomial. Elo solo da "quién es mejor"; el
   mapeo a probabilidades calibradas hay que aprenderlo de los datos.

Este es el PISO. Cualquier modelo más complejo (Dixon-Coles, etc.) tiene que ganarle
a esto en log-loss, o no se justifica.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- Hiperparámetros del Elo (valores estándar tipo eloratings.net) ---
INITIAL_ELO = 1500.0   # rating de arranque para una selección nunca vista
K_FACTOR = 32.0        # cuánto se mueve el rating por partido
HOME_ADV = 65.0        # bonus de Elo para el local cuando NO es campo neutral
OUTCOME_ORDER = ["A", "D", "H"]  # away win, draw, home win (orden ordinal)


def _margin_multiplier(goal_diff: int) -> float:
    """Multiplicador por diferencia de gol (goleadas mueven más el rating).

    Fórmula de World Football Elo Ratings (eloratings.net).
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def compute_elo_history(
    matches: pd.DataFrame,
    k: float = K_FACTOR,
    home_adv: float = HOME_ADV,
    initial: float = INITIAL_ELO,
) -> pd.DataFrame:
    """Devuelve `matches` con columnas nuevas:

    - `elo_home_pre`, `elo_away_pre`: rating de cada equipo ANTES del partido.
    - `elo_diff`: (elo_home_pre - elo_away_pre) + ventaja de localía si no es neutral.
      Esta es la feature que alimenta el modelo de outcome.
    """
    ratings: dict[str, float] = {}
    home_pre = np.empty(len(matches))
    away_pre = np.empty(len(matches))
    diff = np.empty(len(matches))

    home_teams = matches["home_team"].to_numpy()
    away_teams = matches["away_team"].to_numpy()
    home_scores = matches["home_score"].to_numpy()
    away_scores = matches["away_score"].to_numpy()
    neutral = matches["neutral"].to_numpy()

    for i in range(len(matches)):
        h, a = home_teams[i], away_teams[i]
        rh = ratings.get(h, initial)
        ra = ratings.get(a, initial)
        home_pre[i] = rh
        away_pre[i] = ra

        adv = 0.0 if neutral[i] else home_adv
        diff[i] = (rh - ra) + adv

        # Resultado esperado (logístico sobre diferencia de Elo, base 400).
        exp_home = 1.0 / (1.0 + 10 ** (-(rh + adv - ra) / 400.0))

        # Resultado real desde el punto de vista del local.
        gd = home_scores[i] - away_scores[i]
        if gd > 0:
            actual_home = 1.0
        elif gd < 0:
            actual_home = 0.0
        else:
            actual_home = 0.5

        change = k * _margin_multiplier(gd) * (actual_home - exp_home)
        ratings[h] = rh + change
        ratings[a] = ra - change

    out = matches.copy()
    out["elo_home_pre"] = home_pre
    out["elo_away_pre"] = away_pre
    out["elo_diff"] = diff
    return out, ratings


@dataclass
class OutcomeModel:
    """Mapea elo_diff → probabilidades [P(A), P(D), P(H)]."""

    model: LogisticRegression
    classes: list[str]

    @classmethod
    def fit(cls, matches_with_elo: pd.DataFrame) -> "OutcomeModel":
        x = matches_with_elo[["elo_diff"]].to_numpy()
        y = matches_with_elo["outcome"].to_numpy()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x, y)
        return cls(model=clf, classes=list(clf.classes_))

    def predict_proba(self, elo_diff: float | np.ndarray) -> np.ndarray:
        """Devuelve probabilidades en el orden [A, D, H]."""
        arr = np.atleast_2d(np.asarray(elo_diff, dtype=float).reshape(-1, 1))
        proba = self.model.predict_proba(arr)
        # Reordenar columnas a OUTCOME_ORDER por las dudas.
        idx = [self.classes.index(c) for c in OUTCOME_ORDER]
        return proba[:, idx]

    def predict_match(self, elo_home: float, elo_away: float, neutral: bool) -> dict:
        adv = 0.0 if neutral else HOME_ADV
        diff = (elo_home - elo_away) + adv
        p = self.predict_proba(diff)[0]
        return {"away_win": p[0], "draw": p[1], "home_win": p[2]}


def expected_goals(
    elo_home: float, elo_away: float, neutral: bool = True,
    base: float = 1.3, spread: float = 0.8, home_adv: float = HOME_ADV,
) -> tuple[float, float]:
    """Goles esperados (λ) de cada equipo, derivados de la diferencia de Elo.

    Heurística simple (no es un modelo Poisson ajustado — eso es Fase 3): el equipo
    más fuerte sube su λ y el más débil lo baja, manteniendo ~constante el producto.
    """
    adv = 0.0 if neutral else home_adv
    d = (elo_home - elo_away + adv) / 400.0
    lam_home = base * np.exp(spread * d)
    lam_away = base * np.exp(-spread * d)
    return lam_home, lam_away


def _poisson_pmf(k: int, lam: float) -> float:
    import math
    return np.exp(-lam) * lam ** k / math.factorial(k)


def predict_scoreline(
    elo_home: float, elo_away: float, outcome: str,
    neutral: bool = True, max_goals: int = 6,
) -> tuple[int, int]:
    """Marcador más probable COHERENTE con el resultado pronosticado (H/D/A).

    Busca el (goles_local, goles_visitante) más probable bajo Poisson(λ) restringido
    a la región del resultado ya elegido por el modelo, así nunca se contradicen
    (ej. "gana local" + marcador 1-1).
    """
    lam_h, lam_a = expected_goals(elo_home, elo_away, neutral)
    best, best_p = (1, 0), -1.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if outcome == "H" and not i > j:
                continue
            if outcome == "D" and i != j:
                continue
            if outcome == "A" and not i < j:
                continue
            p = _poisson_pmf(i, lam_h) * _poisson_pmf(j, lam_a)
            if p > best_p:
                best_p, best = p, (i, j)
    return best


def _demo() -> None:
    from fetch_data import load_matches

    matches = load_matches()
    with_elo, final_ratings = compute_elo_history(matches)

    # Entrenar el mapeo de outcome con toda la historia.
    model = OutcomeModel.fit(with_elo)

    # Top 15 selecciones por Elo actual.
    top = sorted(final_ratings.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("\n=== Top 15 selecciones por Elo (al último partido del dataset) ===")
    for name, rating in top:
        print(f"  {rating:7.1f}  {name}")

    # Predicción de ejemplo: Argentina vs Francia en campo neutral.
    rating_map = dict(final_ratings)
    examples = [
        ("Argentina", "France", True),
        ("Brazil", "Croatia", True),
        ("Spain", "Morocco", True),
    ]
    print("\n=== Predicciones de ejemplo (campo neutral) ===")
    for home, away, neutral in examples:
        if home not in rating_map or away not in rating_map:
            print(f"  (sin datos para {home} o {away})")
            continue
        pred = model.predict_match(rating_map[home], rating_map[away], neutral)
        print(f"  {home} vs {away}: "
              f"{home} {pred['home_win']:.0%} | empate {pred['draw']:.0%} | "
              f"{away} {pred['away_win']:.0%}")


if __name__ == "__main__":
    _demo()
