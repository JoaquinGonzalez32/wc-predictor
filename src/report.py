"""Genera un mini-reporte explicando POR QUÉ el modelo pronostica un resultado.

Todo sale de datos reales (sin invención): ratings Elo, forma reciente y historial
de enfrentamientos directos del dataset histórico. Se es explícito en que el modelo
se basa en el Elo; la forma y el head-to-head se muestran como contexto.
"""
from __future__ import annotations

import pandas as pd

from baseline_elo import expected_goals


def recent_form(matches: pd.DataFrame, team: str, before: pd.Timestamp, n: int = 5) -> dict:
    """Últimos n partidos del equipo antes de `before` (sin mirar el futuro)."""
    df = matches[((matches["home_team"] == team) | (matches["away_team"] == team))
                 & (matches["date"] < before)].sort_values("date").tail(n)
    w = d = l = 0
    seq = []
    for _, r in df.iterrows():
        gf, ga = (r["home_score"], r["away_score"]) if r["home_team"] == team \
            else (r["away_score"], r["home_score"])
        if gf > ga:
            w += 1; seq.append("✅")
        elif gf < ga:
            l += 1; seq.append("❌")
        else:
            d += 1; seq.append("➖")
    return {"w": w, "d": d, "l": l, "seq": seq, "n": len(df)}


def head_to_head(matches: pd.DataFrame, a: str, b: str, before: pd.Timestamp) -> dict:
    """Historial directo a vs b antes de `before`."""
    df = matches[(((matches["home_team"] == a) & (matches["away_team"] == b))
                  | ((matches["home_team"] == b) & (matches["away_team"] == a)))
                 & (matches["date"] < before)].sort_values("date")
    wa = wb = draws = 0
    for _, r in df.iterrows():
        if r["home_score"] == r["away_score"]:
            draws += 1
            continue
        winner = r["home_team"] if r["home_score"] > r["away_score"] else r["away_team"]
        if winner == a:
            wa += 1
        else:
            wb += 1
    last = None
    if len(df):
        r = df.iloc[-1]
        last = f"{r['home_team']} {int(r['home_score'])}-{int(r['away_score'])} {r['away_team']} ({r['date'].year})"
    return {"n": len(df), "wa": wa, "wb": wb, "draws": draws, "last": last}


def build_report(matches, ratings, m, probs, outcome, score, adv: float = 0.0) -> str:
    """Markdown explicando el pronóstico del partido `m`. `adv` = ventaja de local."""
    t1, t2 = m["team1"], m["team2"]
    e1, e2 = ratings[m["team1_elo"]], ratings[m["team2_elo"]]
    diff = e1 - e2
    before = m["date"] if pd.notna(m["date"]) else matches["date"].max()

    if abs(diff) < 30:
        favor = "Partido **muy parejo** en Elo."
    else:
        stronger, gap = (t1, diff) if diff > 0 else (t2, -diff)
        favor = f"**{stronger}** es favorito por Elo (+{gap:.0f} puntos)."

    f1 = recent_form(matches, m["team1_elo"], before)
    f2 = recent_form(matches, m["team2_elo"], before)
    h = head_to_head(matches, m["team1_elo"], m["team2_elo"], before)

    res_txt = "Empate" if outcome == "D" else f"Gana {t1 if outcome == 'H' else t2}"
    lam1, lam2 = expected_goals(e1, e2, adv=adv)

    lines = [
        f"#### {t1} vs {t2}",
        f"**Resultado previsto:** {res_txt} — marcador más probable **{score[0]}-{score[1]}**",
        "",
        f"**Probabilidades:** {t1} {probs['home_win']:.0%} · "
        f"empate {probs['draw']:.0%} · {t2} {probs['away_win']:.0%}",
        "",
        "**Por qué:**",
        f"- Elo: {t1} {e1:.0f} vs {t2} {e2:.0f}. {favor} El modelo se basa en esta "
        "diferencia para estimar las probabilidades.",
        f"- Goles esperados (derivados de la diferencia de Elo): **{t1} {lam1:.1f}** · "
        f"**{t2} {lam2:.1f}**. El más fuerte tiende a marcar más; de esos valores, el "
        f"marcador entero más probable dentro de «{res_txt.lower()}» es **{score[0]}-{score[1]}**.",
        f"- Forma reciente (últimos {f1['n']}): {t1} {''.join(f1['seq']) or '—'} "
        f"({f1['w']}V {f1['d']}E {f1['l']}D)",
        f"- Forma reciente (últimos {f2['n']}): {t2} {''.join(f2['seq']) or '—'} "
        f"({f2['w']}V {f2['d']}E {f2['l']}D)",
    ]
    if abs(adv) > 0:
        local = t1 if adv > 0 else t2
        lines.append(f"- Ventaja de local: **{local}** juega de anfitrión en su país "
                     f"(+{abs(adv):.0f} de Elo), ya aplicada al pronóstico.")
    if h["n"] > 0:
        lines.append(
            f"- Historial directo: {h['n']} partidos — {t1} {h['wa']} / empates "
            f"{h['draws']} / {t2} {h['wb']}." + (f" Último: {h['last']}." if h["last"] else ""))
    else:
        lines.append("- Historial directo: sin enfrentamientos previos en el dataset.")
    lines.append("")
    lines.append("_Baseline Elo: la forma y el historial son contexto, no entran al "
                 "cálculo. Es un piso de referencia, no un modelo fino._")
    return "\n".join(lines)
