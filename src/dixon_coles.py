"""Fase 3 — Dixon-Coles (modelo de goles).

En vez de mapear Elo→resultado, modela los GOLES de cada equipo con Poisson:

    goles ~ Poisson( exp( ataque_atacante + defensa_defensor + ventaja_local ) )

Se ajusta por máxima verosimilitud como una regresión de Poisson (sklearn) sobre el
histórico, con:
  - parámetros de ataque y defensa por selección,
  - un término de ventaja de local (estimado de los partidos no-neutrales),
  - decaimiento temporal: los partidos viejos pesan menos (half-life configurable),
  - corrección Dixon-Coles τ para marcadores bajos (dependencia 0-0/1-0/0-1/1-1).

De la matriz de marcadores salen las probabilidades 1/X/2 y el marcador más probable.
A diferencia del baseline Elo, esto SÍ predice goles, y se backtestea igual (log-loss).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor


@dataclass
class DixonColes:
    attack: dict[str, float]
    defense: dict[str, float]
    home: float
    intercept: float
    rho: float = -0.05  # corrección de marcadores bajos (fija, valor típico)
    max_goals: int = 10
    teams: list[str] = field(default_factory=list)

    # ---- ajuste ----
    @classmethod
    def fit(cls, matches: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            half_life_days: float = 365 * 3, years_back: int = 12,
            alpha: float = 1e-2, rho: float = -0.05) -> "DixonColes":
        ref = ref_date if ref_date is not None else matches["date"].max()
        df = matches[(matches["date"] <= ref)
                     & (matches["date"] >= ref - pd.Timedelta(days=365 * years_back))].copy()

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Dos observaciones por partido (la de cada equipo anotando).
        rows, cols, vals, y, w = [], [], [], [], []
        age = (ref - df["date"]).dt.days.to_numpy()
        weight = 0.5 ** (age / half_life_days)
        ht = df["home_team"].map(idx).to_numpy()
        at = df["away_team"].map(idx).to_numpy()
        hs = df["home_score"].to_numpy()
        as_ = df["away_score"].to_numpy()
        neu = df["neutral"].to_numpy()

        r = 0
        for k in range(len(df)):
            is_home = 0 if neu[k] else 1
            # obs equipo local anota: ataque=local, defensa=visitante, home=is_home
            rows += [r, r, r]; cols += [ht[k], n + at[k], 2 * n]; vals += [1, 1, is_home]
            y.append(hs[k]); w.append(weight[k]); r += 1
            # obs equipo visitante anota: ataque=visitante, defensa=local, home=0
            rows += [r, r]; cols += [at[k], n + ht[k]]; vals += [1, 1]
            y.append(as_[k]); w.append(weight[k]); r += 1

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, 2 * n + 1))
        model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=500)
        model.fit(X, np.array(y), sample_weight=np.array(w))

        coef = model.coef_
        attack = {t: float(coef[idx[t]]) for t in teams}
        defense = {t: float(coef[n + idx[t]]) for t in teams}
        return cls(attack=attack, defense=defense, home=float(coef[2 * n]),
                   intercept=float(model.intercept_), rho=rho, teams=teams)

    # ---- predicción ----
    def _lambdas(self, t1: str, t2: str, home_for: str | None) -> tuple[float, float]:
        a1, d1 = self.attack.get(t1, 0.0), self.defense.get(t1, 0.0)
        a2, d2 = self.attack.get(t2, 0.0), self.defense.get(t2, 0.0)
        h1 = self.home if home_for == t1 else 0.0
        h2 = self.home if home_for == t2 else 0.0
        lam1 = math.exp(self.intercept + a1 + d2 + h1)
        lam2 = math.exp(self.intercept + a2 + d1 + h2)
        return lam1, lam2

    def _tau(self, i: int, j: int, lam1: float, lam2: float) -> float:
        rho = self.rho
        if i == 0 and j == 0:
            return 1 - lam1 * lam2 * rho
        if i == 0 and j == 1:
            return 1 + lam1 * rho
        if i == 1 and j == 0:
            return 1 + lam2 * rho
        if i == 1 and j == 1:
            return 1 - rho
        return 1.0

    def score_matrix(self, t1: str, t2: str, home_for: str | None = None) -> np.ndarray:
        lam1, lam2 = self._lambdas(t1, t2, home_for)
        mg = self.max_goals
        k = np.arange(mg + 1)
        p1 = np.exp(-lam1) * lam1 ** k / np.array([math.factorial(x) for x in k])
        p2 = np.exp(-lam2) * lam2 ** k / np.array([math.factorial(x) for x in k])
        M = np.outer(p1, p2)
        for i in (0, 1):
            for j in (0, 1):
                M[i, j] *= self._tau(i, j, lam1, lam2)
        return M / M.sum()

    def predict(self, t1: str, t2: str, home_for: str | None = None) -> dict:
        """Probabilidades 1/X/2 (home_win=t1) y goles esperados."""
        M = self.score_matrix(t1, t2, home_for)
        home = float(np.tril(M, -1).sum())  # i>j
        draw = float(np.trace(M))
        away = float(np.triu(M, 1).sum())   # i<j
        lam1, lam2 = self._lambdas(t1, t2, home_for)
        return {"home_win": home, "draw": draw, "away_win": away,
                "lambda_home": lam1, "lambda_away": lam2}

    def scoreline(self, t1: str, t2: str, home_for: str | None = None) -> tuple[int, int]:
        M = self.score_matrix(t1, t2, home_for)
        i, j = np.unravel_index(int(M.argmax()), M.shape)
        return int(i), int(j)


if __name__ == "__main__":
    from fetch_data import load_matches
    m = load_matches()
    dc = DixonColes.fit(m)
    print(f"Ajustado sobre {len(dc.teams)} selecciones · home={dc.home:.3f} "
          f"intercept={dc.intercept:.3f}")
    for a, b in [("Argentina", "France"), ("Brazil", "Croatia"), ("Spain", "Morocco")]:
        p = dc.predict(a, b)
        sc = dc.scoreline(a, b)
        print(f"  {a} vs {b}: {p['home_win']:.0%}/{p['draw']:.0%}/{p['away_win']:.0%} "
              f"· λ {p['lambda_home']:.2f}-{p['lambda_away']:.2f} · marcador {sc[0]}-{sc[1]}")
