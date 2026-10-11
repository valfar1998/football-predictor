# Digest apprendimento — 2026-10-11 05:05 UTC

- Esito fit: **OK**
- Momento: `2026-10-11T05:05:35.912172+00:00`
- Partite chiuse totali: **645**
- Usabili per imparare (ricche): **561** (live 441 + backfill 120)
- Escluse (storico incompleto): **84**

## In cosa sta migliorando / correggendo

- **ROI recente** (ultime 22 giocate, voto ≥8): -33.7% (PnL -0.2877 u)
- **CLV medio:** -0.0003 · beat close +0.0%
- **Soglia EV minima:** da `0.0338` a `0.0426` (più severa, usa anche CLV)
- **Calibrazione probabilità:** aggiornata (8 bin, blend max 0.72).
- **Fattore p online:** `0.9629` (errore medio p−hit 0.0476)
- **Residual EV:** ok su 1805 sample (RMSE 0.478, WF 0.484)
- **Pesi data-signal:** aggiornati (hit rate +40.1%, ROI -15.6%, metodo `walk_forward_brier_roi`).

## Come leggerlo in pratica

- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).
- ROI/CLV **positivi** → filtri un po’ più aperti.
- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.
