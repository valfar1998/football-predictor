# Digest apprendimento — 2026-10-13 05:06 UTC

- Esito fit: **OK**
- Momento: `2026-10-13T05:06:05.500454+00:00`
- Partite chiuse totali: **669**
- Usabili per imparare (ricche): **576** (live 456 + backfill 120)
- Escluse (storico incompleto): **93**

## In cosa sta migliorando / correggendo

- **ROI recente** (ultime 31 giocate, voto ≥8): -27.3% (PnL -0.371 u)
- **CLV medio:** 0.0081 · beat close +6.1%
- **Soglia EV minima:** da `0.0338` a `0.0426` (più severa, usa anche CLV)
- **Calibrazione probabilità:** aggiornata (8 bin, blend max 0.72).
- **Fattore p online:** `0.9872` (errore medio p−hit 0.0164)
- **Residual EV:** ok su 1852 sample (RMSE 0.485, WF 0.502)
- **Pesi data-signal:** aggiornati (hit rate +39.4%, ROI -15.2%, metodo `walk_forward_brier_roi`).

## Come leggerlo in pratica

- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).
- ROI/CLV **positivi** → filtri un po’ più aperti.
- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.
