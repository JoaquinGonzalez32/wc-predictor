"""UI del predictor del Mundial 2026 (Streamlit).

Corré con:
    python -m streamlit run app.py

Muestra los 12 grupos a la vez. Cada grupo se navega por jornada (1/2/3) con flechas.
Los partidos jugados muestran el resultado real; los pendientes muestran, al activar
"Mostrar pronósticos", el resultado previsto + probabilidades. Cada partido tiene un
botón "❓" con un reporte de por qué el modelo llegó a ese pronóstico.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baseline_elo import OutcomeModel, compute_elo_history, predict_scoreline  # noqa: E402
from fetch_data import load_matches  # noqa: E402
from fixture import load_fixture  # noqa: E402
from report import build_report  # noqa: E402


@st.cache_resource(show_spinner="Cargando datos y entrenando el baseline Elo...")
def load_model():
    matches = load_matches()
    with_elo, final_ratings = compute_elo_history(matches)
    model = OutcomeModel.fit(with_elo)
    return model, final_ratings, with_elo, matches["date"].max().date()


@st.cache_data(show_spinner="Bajando el fixture del Mundial 2026...")
def load_fixture_cached() -> pd.DataFrame:
    return load_fixture()


def get_prediction(model, ratings, m):
    """Devuelve (probs, outcome, scoreline) o None si falta algún rating."""
    if not m["is_real"] or m["team1_elo"] not in ratings or m["team2_elo"] not in ratings:
        return None
    p = model.predict_match(ratings[m["team1_elo"]], ratings[m["team2_elo"]], neutral=True)
    outcome = max([("H", p["home_win"]), ("D", p["draw"]), ("A", p["away_win"])],
                  key=lambda kv: kv[1])[0]
    score = predict_scoreline(ratings[m["team1_elo"]], ratings[m["team2_elo"]], outcome)
    return p, outcome, score


def render_match(m, model, ratings, matches, show_pred: bool) -> None:
    with st.container(border=True):
        head, qcol = st.columns([9, 1])
        with head:
            c1, c2, c3 = st.columns([5, 2, 5])
            c1.markdown(f"<div style='text-align:right;font-weight:600'>{m['team1']}</div>",
                        unsafe_allow_html=True)
            c3.markdown(f"<div style='text-align:left;font-weight:600'>{m['team2']}</div>",
                        unsafe_allow_html=True)
            if m["played"]:
                c2.markdown(f"<div style='text-align:center;font-size:1.2rem;font-weight:700'>"
                            f"{int(m['ft1'])}-{int(m['ft2'])}</div>", unsafe_allow_html=True)
            else:
                c2.markdown("<div style='text-align:center;color:#999'>vs</div>",
                            unsafe_allow_html=True)

        pred = None if m["played"] else get_prediction(model, ratings, m)

        # Botón ❓ con el reporte (solo si hay pronóstico que explicar).
        if pred is not None:
            probs, outcome, score = pred
            with qcol:
                with st.popover("❓", use_container_width=True):
                    st.markdown(build_report(matches, ratings, m, probs, outcome, score))

        fecha = m["date"].strftime("%d/%m") if pd.notna(m["date"]) else "?"
        meta = f"📅 {fecha}"
        if m.get("time"):
            meta += f" · 🕐 {m['time']}"
        if m.get("ground"):
            meta += f" · 📍 {m['ground']}"
        st.caption(meta)

        if m["played"]:
            st.caption("✅ Finalizado")
        elif pred is not None and show_pred:
            probs, outcome, score = pred
            t1, t2 = m["team1"], m["team2"]
            res = "Empate" if outcome == "D" else f"Gana {t1 if outcome == 'H' else t2}"
            st.markdown(f"<div style='text-align:center;color:#16a34a;font-weight:700'>"
                        f"🔮 {res} · marcador {score[0]}-{score[1]}</div>",
                        unsafe_allow_html=True)
            pc = st.columns(3)
            for col, (label, prob) in zip(pc, [(t1, probs["home_win"]),
                                               ("Empate", probs["draw"]),
                                               (t2, probs["away_win"])]):
                col.markdown(f"<div style='text-align:center'>"
                             f"<div style='font-size:0.75rem;color:#888'>{label}</div>"
                             f"<div style='font-size:1.1rem;font-weight:600'>{prob:.0%}</div>"
                             f"</div>", unsafe_allow_html=True)


def render_group(group: str, fx: pd.DataFrame, model, ratings, matches, show_pred: bool) -> None:
    st.markdown(f"### {group.replace('Group', 'Grupo')}")
    jkey = f"jornada_{group}"
    jor = st.session_state.setdefault(jkey, 1)

    nl, nc, nr = st.columns([1, 2, 1])
    if nl.button("◀", key=f"prev_{group}", disabled=jor <= 1, use_container_width=True):
        st.session_state[jkey] = jor - 1
        st.rerun()
    nc.markdown(f"<div style='text-align:center;font-weight:600;padding-top:6px'>"
                f"Jornada {jor} <span style='color:#999'>/ 3</span></div>",
                unsafe_allow_html=True)
    if nr.button("▶", key=f"next_{group}", disabled=jor >= 3, use_container_width=True):
        st.session_state[jkey] = jor + 1
        st.rerun()

    sel = fx[(fx["group"] == group) & (fx["jornada"] == jor)].sort_values(["date", "time"])
    for _, m in sel.iterrows():
        render_match(m, model, ratings, matches, show_pred)


def main() -> None:
    st.set_page_config(page_title="WC Predictor 2026", page_icon="⚽", layout="wide")
    st.title("⚽ Mundial 2026 — Predictor")
    st.caption("Probabilidad de cada resultado (gana local / empate / gana visitante) según "
               "el rating Elo. Tocá ❓ en cada partido para ver por qué.")

    model, ratings, matches, last_date = load_model()
    fx = load_fixture_cached()

    show_pred = st.toggle("🔮 Mostrar pronósticos", value=True,
                          help="Apagalo para ver el fixture y resultados sin predicciones.")

    groups = sorted(g for g in fx["group"].unique() if g)
    # Dos columnas de grupos para verlos todos a la vez.
    cols = st.columns(2)
    for i, group in enumerate(groups):
        with cols[i % 2]:
            render_group(group, fx, model, ratings, matches, show_pred)
            st.write("")

    st.divider()
    st.caption(f"Datos hasta {last_date}. Baseline Elo: piso de referencia, no un modelo "
               "fino — en fase final su ventaja sobre el azar es chica (ver README, Fase 3).")


if __name__ == "__main__":
    main()
