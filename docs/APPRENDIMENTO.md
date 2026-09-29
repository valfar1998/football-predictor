# Cosa impara Football Predictor (apprendimento online)

Ogni mattina il job cloud fa due cose diverse:

1. **Riallena il modello base** sullo storico partite (CSV delle leghe).
2. **Apprendimento online** dagli esiti delle giocate archiviate (quando ci sono abbastanza risultati chiusi).

## Cosa aggiorna l’apprendimento online

| Cosa | Effetto pratico |
|---|---|
| **Calibrazione probabilità** | Se il modello è troppo fiducioso/timido, corregge le probabilità. |
| **Soglia EV minima (`min_ev_play`)** | Se il ROI recente (solo voto ≥8) è negativo alza il filtro; se va bene lo allenta un po’. |
| **Fattore probabilità online** | Piccolo moltiplicatore sulle p usate nei filtri. |
| **Residual EV** | Impara quanto l’edge stimato è ottimistico/pessimistico. |
| **Pesi data-signal** | Ribilancia forma, xG, casa/trasferta, ecc. in base a cosa ha funzionato. |

Usa solo partite **chiuse** con dati ricchi (quota + EV + fattori). Lo storico incompleto viene escluso.

## Quando vedi miglioramenti

- **Ogni giorno**: settle esiti + eventuale aggiornamento dei filtri (se ci sono abbastanza sample).
- **Ogni ~7 giorni**: arriva su Telegram un **riassunto** di cosa è cambiato (ROI, CLV, soglie, residual).

Se il report dice “servono ≥25 righe trainable”, l’algoritmo sta ancora raccogliendo esiti: il modello base gira, ma i filtri online non si muovono ancora.

## File tecnici

- `data/models/online_learn_report.json` — ultimo fit (macchina)
- `data/models/learn_digest.md` — stesso contenuto in italiano
- `data/models/calibration.json` — soglie attive
- `data/models/residual_ev.json` / `data_signal_weights.json` — modelli secondari

---

## Ultimo aggiornamento automatico

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
