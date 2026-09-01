# Digest apprendimento — 2026-09-01 05:08 UTC

- Esito fit: **OK**
- Momento: `2026-09-01T05:08:02.621770+00:00`
- Partite chiuse totali: **335**
- Usabili per imparare (ricche): **322** (live 202 + backfill 120)
- Escluse (storico incompleto): **13**

## In cosa sta migliorando / correggendo

- **ROI recente** (ultime 60 giocate): -4.7% (PnL -2.83 u)
- **CLV medio:** -0.0040 · beat close +0.0%
- **Soglia EV minima:** da `0.0305` a `0.036` (più severa, usa anche CLV)
- **Calibrazione probabilità:** aggiornata (8 bin, blend max 0.72).
- **Fattore p online:** `0.9968` (errore medio p−hit 0.0042)
- **Residual EV:** ok su 1692 sample (RMSE 0.469, WF 0.487)
- **Pesi data-signal:** aggiornati (hit rate +41.4%, ROI -7.3%, metodo `walk_forward_brier_roi`).

## Come leggerlo in pratica

- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).
- ROI/CLV **positivi** → filtri un po’ più aperti.
- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.
