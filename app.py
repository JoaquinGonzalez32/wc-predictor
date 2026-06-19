"""UI básica del predictor (Streamlit).

Corré con:
    python -m streamlit run app.py

Dos pestañas:
  - Predicción manual: elegís dos selecciones y muestra 1/X/2.
  - Fixture 2026: calendario real del Mundial con el pronóstico del modelo en los
    partidos que faltan jugar.

Es solo una capa de presentación sobre src/baseline_elo.py + src/fixture.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Hacer importables los módulos de src/.
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baseline_elo import OutcomeModel, compute_elo_history  # noqa: E402
from fetch_data import load_matches  # noqa: E402
from fixture import load_fixture  # noqa: E402


@st.cache_resource(show_spinner="Cargando datos y entrenando el baseline Elo...")
def load_model():
    """Baja datos (si hace falta), calcula Elo y entrena el mapeo de outcome."""
    matches = load_matches()
    with_elo, final_ratings = compute_elo_history(matches)
    model = OutcomeModel.fit(with_elo)
    last_date = matches["date"].max().date()
    return model, final_ratings, last_date


@st.cache_data(show_spinner="Bajando el fixture del Mundial 2026...")
def load_fixture_cached() -> pd.DataFrame:
    return load_fixture()


def render_manual(model, ratings) -> None:
    teams = sorted(ratings.keys())
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo local", teams,
                            index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        default_away = next((t for t in teams if t != home), teams[0])
        away = st.selectbox("Equipo visitante", teams, index=teams.index(default_away))

    neutral = st.checkbox("Campo neutral (sin ventaja de localía)", value=True,
                          help="En un Mundial casi todo es campo neutral, salvo el anfitrión.")

    if home == away:
        st.warning("Elegí dos selecciones distintas.")
        return

    pred = model.predict_match(ratings[home], ratings[away], neutral)
    st.subheader(f"{home} vs {away}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {home}", f"{pred['home_win']:.0%}")
    c2.metric("Empate", f"{pred['draw']:.0%}")
    c3.metric(f"Gana {away}", f"{pred['away_win']:.0%}")
    st.bar_chart(
        {"probabilidad": {
            f"Gana {home}": pred["home_win"],
            "Empate": pred["draw"],
            f"Gana {away}": pred["away_win"],
        }},
        horizontal=True,
    )
    st.caption(
        f"Elo: {home} {ratings[home]:.0f} · {away} {ratings[away]:.0f} · "
        f"diferencia {ratings[home] - ratings[away]:+.0f}"
        + ("" if neutral else " (+65 localía)")
    )

    with st.expander("Ver ranking Elo (top 25)"):
        top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:25]
        st.table({"Selección": [t for t, _ in top], "Elo": [f"{r:.0f}" for _, r in top]})


def render_fixture(model, ratings, fx: pd.DataFrame) -> None:
    # Todos los partidos de Mundial se modelan como campo neutral (simplificación;
    # no se aplica ventaja de localía a los anfitriones).
    groups = sorted(g for g in fx["group"].unique() if g)
    sel = st.selectbox("Filtrar", ["Todos los grupos", *groups])
    view = fx if sel == "Todos los grupos" else fx[fx["group"] == sel]

    upcoming = view[(~view["played"]) & view["is_real"]].sort_values("date")
    played = view[view["played"]].sort_values("date")
    pending_ko = view[~view["is_real"]]

    st.subheader(f"🔮 Próximos partidos ({len(upcoming)})")
    if len(upcoming) == 0:
        st.write("No hay partidos predecibles pendientes en este filtro.")
    else:
        rows = []
        for _, m in upcoming.iterrows():
            p = model.predict_match(ratings[m["team1_elo"]], ratings[m["team2_elo"]], True)
            fav = max(
                ((m["team1"], p["home_win"]), ("Empate", p["draw"]), (m["team2"], p["away_win"])),
                key=lambda kv: kv[1],
            )
            rows.append({
                "Fecha": m["date"].date() if pd.notna(m["date"]) else "",
                "Grupo": m["group"],
                "Local": m["team1"],
                "Visitante": m["team2"],
                f"P(Local)": p["home_win"],
                "P(Empate)": p["draw"],
                f"P(Visit)": p["away_win"],
                "Favorito": f"{fav[0]} ({fav[1]:.0%})",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df, hide_index=True, use_container_width=True,
            column_config={
                "P(Local)": st.column_config.ProgressColumn(
                    "P(Local)", format="%.0f%%", min_value=0, max_value=1),
                "P(Empate)": st.column_config.ProgressColumn(
                    "P(Empate)", format="%.0f%%", min_value=0, max_value=1),
                "P(Visit)": st.column_config.ProgressColumn(
                    "P(Visit)", format="%.0f%%", min_value=0, max_value=1),
            },
        )

    with st.expander(f"✅ Resultados ya jugados ({len(played)})"):
        if len(played) == 0:
            st.write("Todavía no hay partidos jugados en este filtro.")
        else:
            res = pd.DataFrame({
                "Fecha": [d.date() if pd.notna(d) else "" for d in played["date"]],
                "Grupo": played["group"].values,
                "Partido": [f"{a} {int(g1)}-{int(g2)} {b}" for a, g1, g2, b in zip(
                    played["team1"], played["ft1"], played["ft2"], played["team2"])],
            })
            st.dataframe(res, hide_index=True, use_container_width=True)

    if len(pending_ko) > 0:
        st.caption(
            f"🔒 {len(pending_ko)} partidos de eliminatoria todavía sin equipos definidos "
            "(se predicen cuando se conozcan los clasificados → Fase 5: simulación del bracket)."
        )


def main() -> None:
    st.set_page_config(page_title="WC Predictor", page_icon="⚽", layout="centered")
    st.title("⚽ WC Predictor")
    st.caption("Baseline Elo — probabilidades 1/X/2 por partido")

    model, ratings, last_date = load_model()
    fx = load_fixture_cached()
    st.info(f"Datos hasta **{last_date}** · {len(ratings):,} selecciones con rating Elo")

    tab_fx, tab_manual = st.tabs(["🏆 Fixture Mundial 2026", "🎯 Predicción manual"])
    with tab_fx:
        render_fixture(model, ratings, fx)
    with tab_manual:
        render_manual(model, ratings)

    st.divider()
    st.caption(
        "⚠️ Baseline Elo: piso de referencia, no un modelo fino. En fase final de "
        "Mundial su ventaja sobre el azar es chica — ver Fase 3 (Dixon-Coles) en el README."
    )


if __name__ == "__main__":
    main()
