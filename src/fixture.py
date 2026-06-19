"""Fixture del Mundial 2026.

Baja el calendario oficial (104 partidos, 12 grupos + eliminatorias) del dataset
public-domain openfootball/worldcup.json — sin API key. Lo cachea en data/ y lo
normaliza a un DataFrame, reconciliando los nombres de equipo con los del modelo Elo.

Los partidos de eliminatorias traen placeholders (W101, 1A, 2B...) hasta que se
conocen los clasificados: esos NO son predecibles todavía (columna `is_real=False`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

FIXTURE_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIXTURE_JSON = DATA_DIR / "wc2026_fixture.json"

# Nombres del fixture (openfootball) -> nombres del dataset de resultados (martj42).
NAME_ALIASES = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def canon(name: str) -> str:
    """Nombre canónico usado por el modelo Elo."""
    return NAME_ALIASES.get(name, name)


def _is_placeholder(team: str) -> bool:
    """True si es un placeholder de eliminatoria (W101, 1A, 2B, RU-A...), no un equipo real."""
    return bool(team) and any(ch.isdigit() for ch in team)


def download_fixture(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if FIXTURE_JSON.exists() and not force:
        return FIXTURE_JSON
    resp = requests.get(FIXTURE_URL, timeout=60)
    resp.raise_for_status()
    FIXTURE_JSON.write_bytes(resp.content)
    return FIXTURE_JSON


def load_fixture(force_download: bool = False) -> pd.DataFrame:
    """DataFrame del fixture, una fila por partido.

    Columnas:
      num, round, group, date, ground, team1, team2,
      team1_elo, team2_elo  -> nombres canónicos para el modelo
      is_real               -> ambos equipos son selecciones reales (predecible)
      played                -> ya tiene resultado final
      ft1, ft2              -> goles finales (NaN si no se jugó)
    """
    import json

    path = download_fixture(force=force_download)
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for i, m in enumerate(data["matches"]):
        t1, t2 = m.get("team1", ""), m.get("team2", "")
        is_real = not _is_placeholder(t1) and not _is_placeholder(t2) and bool(t1) and bool(t2)
        score = m.get("score") or {}
        ft = score.get("ft")
        played = isinstance(ft, list) and len(ft) == 2
        rows.append({
            "num": m.get("num", i + 1),
            "round": m.get("round", ""),
            "group": m.get("group", ""),
            "date": m.get("date", ""),
            "time": m.get("time", ""),
            "ground": m.get("ground", ""),
            "team1": t1,
            "team2": t2,
            "team1_elo": canon(t1) if is_real else None,
            "team2_elo": canon(t2) if is_real else None,
            "is_real": is_real,
            "played": played,
            "ft1": ft[0] if played else None,
            "ft2": ft[1] if played else None,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _assign_jornada(df)
    return df


def _assign_jornada(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna jornada 1/2/3 DENTRO de cada grupo.

    Un grupo de 4 equipos juega 3 jornadas de 2 partidos cada una (todos los equipos
    juegan una vez por jornada). El 'round' del dataset ("Matchday N") es global al
    torneo y NO sirve como jornada: a veces una jornada de un grupo se reparte en dos
    matchdays globales (p.ej. Grupo B: rounds 2,3,8,14), lo que rompía el mapeo 1-round-
    1-jornada y dejaba partidos fuera.

    Solución robusta: ordenar los partidos del grupo cronológicamente y agruparlos de a
    dos (cada par = una jornada completa con los 4 equipos).
    """
    df["jornada"] = pd.NA
    gs = df[df["round"].str.startswith("Matchday", na=False)]
    for _, sub in gs.groupby("group"):
        order = sub.sort_values(["date", "num"]).index
        for pos, idx in enumerate(order):
            df.loc[idx, "jornada"] = pos // 2 + 1
    df["jornada"] = df["jornada"].astype("Int64")
    return df


def is_group_stage(round_name: str) -> bool:
    return round_name.startswith("Matchday")


# Sedes del Mundial 2026 por país anfitrión (México y Canadá listadas; el resto = USA).
_MEXICO_CITIES = ("Mexico City", "Guadalajara", "Zapopan", "Monterrey", "Guadalupe")
_CANADA_CITIES = ("Toronto", "Vancouver")


def host_nation(ground: str) -> str | None:
    """País anfitrión donde se juega el partido, según la sede. None si no hay sede."""
    if not ground:
        return None
    if any(c in ground for c in _MEXICO_CITIES):
        return "Mexico"
    if any(c in ground for c in _CANADA_CITIES):
        return "Canada"
    return "USA"  # las 11 sedes restantes del torneo son en Estados Unidos


def host_advantage(team1: str, team2: str, ground: str, home_adv: float = 65.0) -> float:
    """Bonus de Elo para el local (team1) si un anfitrión juega en su país.

    +home_adv si team1 es el anfitrión local, -home_adv si lo es team2, 0 si neutral.
    """
    host = host_nation(ground)
    if host is None:
        return 0.0
    if team1 == host:
        return home_adv
    if team2 == host:
        return -home_adv
    return 0.0


if __name__ == "__main__":
    fx = load_fixture(force_download=True)
    print(f"{len(fx)} partidos · {fx['group'].replace('', pd.NA).dropna().nunique()} grupos")
    print(f"Jugados: {fx['played'].sum()} · Predecibles pendientes: "
          f"{((~fx['played']) & fx['is_real']).sum()} · "
          f"Con placeholder: {(~fx['is_real']).sum()}")
    print("\nPróximos 5 partidos predecibles:")
    upcoming = fx[(~fx["played"]) & fx["is_real"]].head(5)
    print(upcoming[["date", "group", "team1", "team2"]].to_string(index=False))
