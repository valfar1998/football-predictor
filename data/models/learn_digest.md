# Digest apprendimento — 2026-10-01 11:28 UTC

- Esito fit: **OK**
- Momento: `2026-10-01T11:28:30.167168+00:00`
- Partite chiuse totali: **642**
- Usabili per imparare (ricche): **562** (live 442 + backfill 120)
- Escluse (storico incompleto): **80**

## In cosa sta migliorando / correggendo

- **ROI recente** (ultime 24 giocate, voto ≥8): -5.0% (PnL -1.2 u)
- **CLV medio:** 0.0008 · beat close +0.0%
- **Soglia EV minima:** da `0.0305` a `0.036` (più severa, usa anche CLV)
- **Calibrazione probabilità:** aggiornata (8 bin, blend max 0.72).
- **Fattore p online:** `0.9622` (errore medio p−hit 0.0485)
- **Residual EV:** ok su 2207 sample (RMSE 0.478, WF 0.484)
- **Pesi data-signal:** aggiornati (hit rate +39.8%, ROI -15.8%, metodo `walk_forward_brier_roi`).

## Come leggerlo in pratica

- ROI/CLV **negativi** → filtri più stretti (meno giocate dubbie).
- ROI/CLV **positivi** → filtri un po’ più aperti.
- Residual/pesi → correggono edge e analisi dati, non riscrivono il modello ML base.
