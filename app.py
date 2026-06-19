"""UI del predictor del Mundial 2026 (Streamlit).

Corré con:
    python -m streamlit run app.py

Una sola vista: elegís un grupo y navegás sus 3 jornadas con flechas. Cada jornada
muestra sus partidos con fecha/horario; los ya jugados muestran el resultado y los
pendientes se pueden predecir con un botón.

El modelo (baseline Elo) estima, para cada partido, la PROBABILIDAD de cada resultado
posible: que gane el local, que sea empate, o que gane el visitante. Esas tres
probabilidades suman 100%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baseline_elo import OutcomeModel, compute_elo_history  # noqa: E402
from fetch_data import load_matches  # noqa: E402
from fixture import load_fixture  # noqa: E402


@st.cache_resource(show_spinner="Cargando datos y entrenando el baseline Elo...")
def load_model():
    matches = load_matches()
    with_elo, final_ratings = compute_elo_history(matches)
    model = OutcomeModel.fit(with_elo)
    return model, final_ratings, matches["date"].max().date()


@st.cache_data(show_spinner="Bajando el fixture del Mundial 2026...")
def load_fixture_cached() -> pd.DataFrame:
    return load_fixture()


def predict(model, ratings, m) -> dict | None:
    """Probabilidades 1/X/2 de un partido (campo neutral). None si falta algún rating."""
    if not m["is_real"] or m["team1_elo"] not in ratings or m["team2_elo"] not in ratings:
        return None
    return model.predict_match(ratings[m["team1_elo"]], ratings[m["team2_elo"]], neutral=True)


def render_match(m, model, ratings, show_prediction: bool) -> None:
    """Una tarjeta de partido."""
    with st.container(border=True):
        c1, c2, c3 = st.columns([5, 2, 5])
        c1.markdown(f"<div style='text-align:right;font-size:1.1rem;font-weight:600'>"
                    f"{m['team1']}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='text-align:left;font-size:1.1rem;font-weight:600'>"
                    f"{m['team2']}</div>", unsafe_allow_html=True)

        if m["played"]:
            c2.markdown(
                f"<div style='text-align:center;font-size:1.3rem;font-weight:700'>"
                f"{int(m['ft1'])} - {int(m['ft2'])}</div>", unsafe_allow_html=True)
        else:
            c2.markdown("<div style='text-align:center;color:#888'>vs</div>",
                        unsafe_allow_html=True)

        fecha = m["date"].strftime("%d/%m/%Y") if pd.notna(m["date"]) else "?"
        meta = f"📅 {fecha}"
        if m.get("time"):
            meta += f" · 🕐 {m['time']}"
        if m.get("ground"):
            meta += f" · 📍 {m['ground']}"
        st.caption(meta)

        if m["played"]:
            st.caption("✅ Finalizado")
        elif show_prediction:
            p = predict(model, ratings, m)
            if p is None:
                st.caption("Sin datos suficientes para predecir.")
                return
            opts = [(m["team1"], p["home_win"]), ("Empate", p["draw"]),
                    (m["team2"], p["away_win"])]
            fav = max(opts, key=lambda kv: kv[1])
            pc = st.columns(3)
            for col, (label, prob) in zip(pc, opts):
                is_fav = label == fav[0]
                color = "#16a34a" if is_fav else "#666"
                weight = "700" if is_fav else "500"
                col.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:0.8rem;color:#888'>{label}</div>"
                    f"<div style='font-size:1.4rem;font-weight:{weight};color:{color}'>"
                    f"{prob:.0%}</div></div>", unsafe_allow_html=True)
            st.progress(fav[1], text=f"🔮 Pronóstico: **{fav[0]}** ({fav[1]:.0%})")


def main() -> None:
    st.set_page_config(page_title="WC Predictor 2026", page_icon="⚽", layout="centered")
    st.title("⚽ Mundial 2026 — Predictor")
    st.caption("Probabilidad de cada resultado (gana local / empate / gana visitante) "
               "según el rating Elo de cada selección.")

    model, ratings, last_date = load_model()
    fx = load_fixture_cached()

    groups = sorted(g for g in fx["group"].unique() if g)
    group = st.selectbox("Grupo", groups, format_func=lambda g: g.replace("Group", "Grupo"))

    # Navegación de jornada con flechas (estado por grupo).
    jkey = f"jornada_{group}"
    if jkey not in st.session_state:
        st.session_state[jkey] = 1
    jor = st.session_state[jkey]

    nav_l, nav_c, nav_r = st.columns([1, 3, 1])
    if nav_l.button("◀", disabled=jor <= 1, use_container_width=True):
        st.session_state[jkey] = jor - 1
        st.rerun()
    nav_c.markdown(
        f"<div style='text-align:center;font-size:1.2rem;font-weight:600;padding-top:4px'>"
        f"Jornada {jor} <span style='color:#888;font-size:0.9rem'>de 3</span></div>",
        unsafe_allow_html=True)
    if nav_r.button("▶", disabled=jor >= 3, use_container_width=True):
        st.session_state[jkey] = jor + 1
        st.rerun()

    # Partidos de este grupo + jornada.
    sel = fx[(fx["group"] == group) & (fx["jornada"] == jor)].sort_values(["date", "time"])

    # Botón de predecir (solo si hay pendientes predecibles).
    pendientes = sel[(~sel["played"]) & sel["is_real"]]
    pred_key = f"pred_{group}_{jor}"
    if len(pendientes) > 0:
        if st.button(f"🔮 Predecir jornada {jor}", type="primary", use_container_width=True):
            st.session_state[pred_key] = True
    show_pred = st.session_state.get(pred_key, False)

    st.write("")
    for _, m in sel.iterrows():
        render_match(m, model, ratings, show_pred)

    st.divider()
    st.caption(f"Datos hasta {last_date}. Baseline Elo: piso de referencia, no un modelo "
               "fino — en fase final su ventaja sobre el azar es chica (ver README, Fase 3).")


if __name__ == "__main__":
    main()
