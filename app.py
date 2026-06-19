"""UI básica del predictor (Streamlit).

Corré con:
    streamlit run app.py

Elegís dos selecciones + si es campo neutral, y muestra las probabilidades 1/X/2
del baseline Elo. Es solo una capa de presentación sobre src/baseline_elo.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Hacer importables los módulos de src/.
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baseline_elo import OutcomeModel, compute_elo_history  # noqa: E402
from fetch_data import load_matches  # noqa: E402


@st.cache_resource(show_spinner="Cargando datos y entrenando el baseline Elo...")
def load_model():
    """Baja datos (si hace falta), calcula Elo y entrena el mapeo de outcome.

    Cacheado: corre una sola vez por sesión del server.
    """
    matches = load_matches()
    with_elo, final_ratings = compute_elo_history(matches)
    model = OutcomeModel.fit(with_elo)
    last_date = matches["date"].max().date()
    return model, final_ratings, last_date


def main() -> None:
    st.set_page_config(page_title="WC Predictor", page_icon="⚽", layout="centered")
    st.title("⚽ WC Predictor")
    st.caption("Baseline Elo — probabilidades 1/X/2 por partido")

    model, ratings, last_date = load_model()
    teams = sorted(ratings.keys())

    st.info(f"Datos hasta **{last_date}** · {len(teams):,} selecciones con rating Elo")

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo local", teams,
                            index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        # Default razonable para el segundo: el primero distinto del local.
        default_away = next((t for t in teams if t != home), teams[0])
        away = st.selectbox("Equipo visitante", teams, index=teams.index(default_away))

    neutral = st.checkbox("Campo neutral (sin ventaja de localía)", value=True,
                          help="En un Mundial casi todo es campo neutral, salvo el anfitrión.")

    if home == away:
        st.warning("Elegí dos selecciones distintas.")
        return

    pred = model.predict_match(ratings[home], ratings[away], neutral)
    elo_home, elo_away = ratings[home], ratings[away]

    st.subheader(f"{home} vs {away}")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gana {home}", f"{pred['home_win']:.0%}")
    c2.metric("Empate", f"{pred['draw']:.0%}")
    c3.metric(f"Gana {away}", f"{pred['away_win']:.0%}")

    # Barra visual simple.
    st.bar_chart(
        {"probabilidad": {
            f"Gana {home}": pred["home_win"],
            "Empate": pred["draw"],
            f"Gana {away}": pred["away_win"],
        }},
        horizontal=True,
    )

    st.caption(
        f"Elo: {home} {elo_home:.0f} · {away} {elo_away:.0f} · "
        f"diferencia {elo_home - elo_away:+.0f}"
        + ("" if neutral else " (+65 localía)")
    )

    with st.expander("Ver ranking Elo (top 25)"):
        top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:25]
        st.table({"Selección": [t for t, _ in top],
                  "Elo": [f"{r:.0f}" for _, r in top]})

    st.divider()
    st.caption(
        "⚠️ Baseline Elo: piso de referencia, no un modelo fino. En fase final de "
        "Mundial su ventaja sobre el azar es chica — ver Fase 3 (Dixon-Coles) en el README."
    )


if __name__ == "__main__":
    main()
