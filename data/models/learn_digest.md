# Digest apprendimento — 2026-09-29 05:04 UTC

- Esito fit: **OK**
- Momento: `2026-09-29T05:04:47.566091+00:00`
- Partite chiuse totali: **622**
- Usabili per imparare (ricche): **560** (live 440 + backfill 120)
- Escluse (storico incompleto): **62**

## In cosa sta migliorando / correggendo

- **ROI recente:** non ancora calcolabile (pochi esiti live con voto ≥8).
- **CLV medio:** 0.0007 · beat close +0.0%
- **Soglia EV minima:** da `0.0272` a `0.0294` (più severa, usa anche CLV)
- **Calibrazione probabilità:** aggiornata (8 bin, blend max 0.72).
- **Fattore p online:** `0.9629` (errore medio p−hit 0.0476)
- **Residual EV:** ok su 2195 sample (RMSE 0.478, WF 0.485)
- **Pesi data-signal:** aggiornati (hit rate +39.8%, ROI -15.9%, metodo `walk_forward_brier_roi`).

## Come leggerlo in pratica

- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).
- ROI/CLV **positivi** → filtri un po’ più aperti.
- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.
