# Roadmap — stato e prossimi passi

Allineato a `PROJECT_BRIEF.md`. **Aggiornato: 2026-08-22** (dopo backfill synthetic + learn trainable-only).

**Regola d’oro:** live/contesto fragile → quadro/voto/no_bet; EV/Kelly solo da modelli/calibrazione su OOF o settled.

**Apprendimento online:** fit solo su righe **trainable** (riche). Le ~679 live vecchie senza quota/EV/fattori sono **ignorate**. Live ricche ×5 + backfill synthetic ×4 nel fit.

**Convenzione:** dopo ogni modifica al codice, aggiornare **questo file** e `PROJECT_BRIEF.md`.

---

## Snapshot operativo (SQLite `our_history.sqlite`)

| Gate | Target | **Ora** | Stato |
|------|--------|---------|--------|
| Settled totali | ≥ 30 (peso storico voto) | **830** | ✅ |
| **Trainable** (riche live + backfill) | ≥ 80 | **151** | ✅ |
| — live ricche | — | **51** | 🟡 cresce con Solo quote |
| — backfill synthetic | — | **100** | ✅ bootstrap |
| Residual WF-RMSE | ≤ 0,55 | **≈ 0,53** | ✅ produzione |
| `online_p_factor` | live ricche ≥30 | **30** (factor 1.005) | ✅ |
| Pesi `data_signal` | n trainable fit | **655** (151×repl) | ✅ |
| `quota_pick` paper live | quanto più alto | **51 live** | 🟡 |

**Storico ricco** = riga settled archiviata **prima** del match con tutti: `quota_pick`, `ev_cons` o `ev_sharp`, `data_factors`, `agree_share`.

**Puoi andare avanti col progetto?** **Sì** — residual, pesi e calibrazione online usano **151 righe trainable** (100 backfill + 51 live ricche). Lo storico live incompleto non inquina più il fit. Il prossimo passo operativo è far crescere le **live ricche** con refresh pre-match weekend.

---

## Automazione calendario (per alimentare lo storico)

| Canale | Effetto |
|--------|---------|
| UI **Aggiorna dati + modello** / **Solo quote** / `--update` / `--odds-update` | `archive_upcoming` + `settle_pending` |
| Task Scheduler + `scripts/notify_refresh_hidden.vbs` | refresh locale senza UI |
| GHA `Aggiorna dati e modello` (05:00 UTC ≈ 07:00 IT) | train + calendario + archive/settle/**learn** + Telegram |
| GHA Telegram ogni 30′ | solo alert (usa modello/calendario in cache) |
| GHA `Ritreno settimanale` (dom 04:00 UTC) | train pesante + artefatti `best_model` + `market_models` |
| UI / CLI **Scarica modello da GitHub** (`--pull-model`) | modello cloud in locale (no train 1h+) |

**Per far crescere lo storico ricco:** ogni giorno (o almeno prima delle partite che segui) lancia **Solo quote e calendario** o **Aggiorna dati + modello** così `archive_upcoming` salva EV/quote/fattori **pre-match**; dopo il risultato, `settle_pending` chiude la riga e **Apprendi da partite chiuse** aggiorna calibrazione/residual/pesi.

---

## Aperti (operatività / dati)

### 1) Live ricche — crescita continua (51 → 80+ live-only)

- **Backfill synthetic (100)** → bootstrap fatto; non sostituisce il live ma accelera bins/residual/pesi.
- **Cosa manca:** più archivi **live** pre-match completi (quota+EV+fattori) su Top leghe.
- **Azione:** **Solo quote** prima del KO; preferire partite con quote reali / Asian.

### 2) Residual EV in produzione stabile — ✅ FATTO

- Codice attivo: `mode: full_production`, n=620 fit, WF-RMSE ≈ 0,53 (trainable-only + replicate).
- Verifica in UI tab **Valutazione** dopo **Apprendi da partite chiuse**.

### 3) Paper Kelly / ROI @ quote reali — 🟡 parziale

- **51 live** con `quota_pick`; backfill non ha quote reali pre-match.
- Migliora con più archivi live ricchi pre-match.

### 4) Pesi `data_signal` su ROI reale — ✅ operativo

- `optimize_weights` su 655 righe replicate (151 trainable); Brier ≈ 0,25.
- Bottone **Ottimizza pesi Analisi dati** in Valutazione (dopo Apprendi).

---

## Chiusi di recente (non ripetere)

- Progresso `calendario N/M` durante **Aggiorna dati + modello** (95→99%, non più barra ferma).
- **Bottoni quote leggeri:** `refresh_upcoming_odds` (EV/Kelly senza ML/MC); `build_upcoming` riusa predizioni; contesto skip se cache <72h; notify-refresh light.
- Progresso % + log live sui bottoni lunghi (UI + echo terminale).
- **Cards/corners fonti gratis:** FD `HY/AY/HC/AC` → `fd_side_rates.csv`; FBref match logs opzionale.
- **Marcatori:** Understat player xG + FBref; lineup FotMob su top picks.
- Cluster ML, XGB O/U+AH, conformal, residual/pesi WF, history ricca, online_learn, ritreno GHA, …
- **No-bet ragionevole:** conformal width 0.70 non è più veto; O/U–AH solo sul mercato del pick; bin OOF protetti; `online_p_factor` 1X2 n≥80, floor 0.96.
- Warning soccerdata (stagione `2021`, concat pandas, Mondiali vs club) e sklearn `y_prob` non somma a 1: stagioni `2526`, FBref solo Big 5, p rinormalizzate.
- **UI copia voto:** su singola / calendario / mercati, testo + scheda grafica in clipboard o download PNG/txt.
- **Solo quote fermo a 0%:** progresso in-process + `python -u` sugli CLI figli.
- **Score pro:** pesi fonti, Unified+Confidence+Risk 0–100, Priorità calendario, override meteo/assenze, Bet Type Recommender (`pro_scores.py`).

---

## Ordine consigliato (da ora)

```text
1. Refresh pre-match quotidiano (Solo quote o Aggiorna) → storico ricco 51→80+
2. Apprendi da partite chiuse ogni 1–2 giorni
3. Monitor Valutazione: residual (ok), paper ROI, pesi data_signal
4. Quando online_p_factor ≥80 settled 1X2: controllare calibrazione live vs OOF
```

---

*Item chiuso → spostarlo in “Chiusi di recente” e aggiornare la checklist in `PROJECT_BRIEF.md`.*
