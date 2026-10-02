# Digest apprendimento — 2026-10-02 05:07 UTC

- Esito fit: **OK**
- Momento: `2026-10-02T05:07:45.718135+00:00`
- Partite chiuse totali: **640**
- Usabili per imparare (ricche): **560** (live 440 + backfill 120)
- Escluse (storico incompleto): **80**

## In cosa sta migliorando / correggendo

- **ROI recente** (ultime 24 giocate, voto ≥8): -5.0% (PnL -1.2 u)
- **CLV medio:** 0.0007 · beat close +0.0%
- **Soglia EV minima:** da `0.0305` a `0.036` (più severa, usa anche CLV)
- **Calibrazione probabilità:** aggiornata (8 bin, blend max 0.72).
- **Fattore p online:** `0.9629` (errore medio p−hit 0.0476)
- **Residual EV:** ok su 2195 sample (RMSE 0.478, WF 0.485)
- **Pesi data-signal:** aggiornati (hit rate +39.8%, ROI -15.9%, metodo `walk_forward_brier_roi`).

## Come leggerlo in pratica

- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).
- ROI/CLV **positivi** → filtri un po’ più aperti.
- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.
