# wc-predictor

Predictor de resultados de partidos (1/X/2) para el Mundial, empezando simple y validando.

Filosofía: **un baseline honesto primero**. Cualquier modelo más complejo tiene que
ganarle al baseline Elo medido con log-loss/Brier — si no le gana, no se agrega.

## Estado

- **Fase 1 — Datos** ✅ `src/fetch_data.py`
  Baja `results.csv` del dataset [martj42/international_results](https://github.com/martj42/international_results)
  (resultados de selecciones desde 1872, CSV directo, sin scraping). Cachea en `data/`.
- **Fase 2 — Baseline Elo** ✅ `src/baseline_elo.py`
  Calcula ratings Elo de cada selección desde cero recorriendo la historia de partidos,
  y mapea la diferencia de Elo a probabilidades 1/X/2 con una regresión ordenada.
  `src/backtest.py` mide log-loss y Brier contra un baseline trivial (tasas base).

## Próximas fases (no implementadas todavía)

- Fase 3 — Dixon-Coles (Poisson bivariado sobre goles).
- Fase 4 — Backtesting formal contra cuotas de casas de apuestas.
- Fase 5 — Simulación Monte Carlo del bracket 2026.
- Fase 6 — Fuerza de selección derivada del rendimiento de jugadores en clubes (solo si hace falta).

## Setup (Windows / PowerShell)

```powershell
cd C:/dev/wc-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1     # PowerShell (NO 'activate' suelto)
pip install -r requirements.txt
```

> Si `Activate.ps1` falla por execution policy:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (una vez).
> El venv es opcional: si ya tenés las deps en el Python global, podés saltearlo.

## Uso

```powershell
# UI (lo más cómodo): abre la app en el browser
python -m streamlit run app.py

# O por línea de comandos:
python src/fetch_data.py     # 1. bajar datos (cachea en data/results.csv)
python src/baseline_elo.py   # 2. entrenar baseline + predicciones de ejemplo
python src/backtest.py       # 3. backtest: log-loss/Brier vs baseline trivial
```

## Cómo se evalúa

No por accuracy seca (un modelo de probabilidades se juzga por **calibración**):

- **Log-loss** — penaliza fuerte estar seguro y equivocado. Métrica principal.
- **Brier score** — error cuadrático medio sobre las probabilidades.

El benchmark a vencer, en orden: (1) tasas base, (2) Elo, (3) cuotas de las casas.

## Notas de datos

- Fuente: martj42/international_results — `date, home_team, away_team, home_score,
  away_score, tournament, city, country, neutral`.
- `neutral=True` → no se aplica ventaja de localía (clave en Mundiales: casi todo es
  campo neutral salvo el anfitrión).
