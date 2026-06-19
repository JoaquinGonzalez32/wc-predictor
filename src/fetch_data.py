"""Fase 1 — Datos.

Baja el dataset de resultados de selecciones (martj42/international_results) a data/.
Es un CSV directo en GitHub: sin scraping, sin rate-limits, sin API keys.

Columnas: date, home_team, away_team, home_score, away_score, tournament, city,
country, neutral.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# Repo de Mart Jürisoo: resultados de partidos internacionales desde 1872.
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"


def download(force: bool = False) -> Path:
    """Descarga results.csv a data/. Si ya existe y no se fuerza, no re-baja."""
    DATA_DIR.mkdir(exist_ok=True)
    if RESULTS_CSV.exists() and not force:
        print(f"[cache] {RESULTS_CSV} ya existe — usá force=True para re-bajar.")
        return RESULTS_CSV

    print(f"[fetch] bajando {RESULTS_URL} ...")
    resp = requests.get(RESULTS_URL, timeout=60)
    resp.raise_for_status()
    RESULTS_CSV.write_bytes(resp.content)
    print(f"[ok] guardado en {RESULTS_CSV} ({len(resp.content) / 1024:.0f} KB)")
    return RESULTS_CSV


def load_matches(force_download: bool = False) -> pd.DataFrame:
    """Carga los partidos como DataFrame, tipado y ordenado por fecha.

    Devuelve solo partidos con resultado conocido (descarta futuros/sin score).
    Agrega columna `outcome` ∈ {'H', 'D', 'A'} (home win / draw / away win).
    """
    path = download(force=force_download)
    df = pd.read_csv(path, parse_dates=["date"])

    # Normalizar tipos.
    df["neutral"] = df["neutral"].astype(bool)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Resultado categórico.
    df["outcome"] = "D"
    df.loc[df["home_score"] > df["away_score"], "outcome"] = "H"
    df.loc[df["home_score"] < df["away_score"], "outcome"] = "A"

    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    matches = load_matches(force_download=True)
    print(f"\n{len(matches):,} partidos desde {matches['date'].min().date()} "
          f"hasta {matches['date'].max().date()}")
    print("\nDistribución de resultados (todos los partidos):")
    print(matches["outcome"].value_counts(normalize=True).round(3))
    print("\nÚltimos 5 partidos:")
    print(matches.tail(5)[["date", "home_team", "away_team",
                           "home_score", "away_score", "tournament"]].to_string(index=False))
