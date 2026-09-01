# Roadmap — stato e prossimi passi

Allineato a `PROJECT_BRIEF.md`. **Aggiornato: 2026-08-30** (settle secondari, phasing-out backfill, GHA odds-prefresh).

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

## Aperti (operatività / dati) — verso 10/10

### 1) Live ricche — **51 → 80+** (priorità operativa)

- Routine pre-match: **Solo quote** / `--odds-update` ogni giorno.
- **GHA `odds-prefresh.yml`**: 10:00 + 16:00 UTC (archivia pre-KO).
- 80+ fixture con `quota_pick` + EV + fattori → paper/ROI indipendente dal backfill.

### 2) Settle mercati secondari — ✅ FATTO (codice)

- Cards/corners: FD `HY/AY/HR/AR/HC/AC` in `settle_from_results`.
- Scorer: `settle_scorer_pending()` + FotMob `extract_goal_scorers` + fuzzy match.

### 3) Connettori dati — 🟡 migliorato

- Betfair 403 CI: soft-fail + **cache stale** ✅; `http_client` UA rotation ✅.
- Asian timeout GHA: retry + skip ✅.
- FotMob/FBref: cache + monitor rotture.

### 4) Phasing-out backfill synthetic — ✅ gate automatico @150 live

- `learn_policy.backfill_excluded_from_fit`: esclude synthetic da `replicate_for_fit`.

### 5) Residual EV in produzione — ✅ FATTO

- WF-RMSE ≈ 0,53; verifica tab Valutazione.

### 6) Pesi `data_signal` — ✅ operativo

- Walk-forward su trainable; Ottimizza pesi in UI dopo Apprendi.

---

## Chiusi (codice / recenti)

- Progresso `calendario N/M` durante **Aggiorna dati + modello** (95→99%, non più barra ferma).
- **Bottoni quote leggeri:** `refresh_upcoming_odds` (EV/Kelly senza ML/MC); **Solo quote** = fd + Asian + Pinnacle/Betfair cache, niente mondiale/coppe/tipster/FBref; notify-refresh light.
- Progresso % + log live sui bottoni lunghi (UI + echo terminale).
- **Cards/corners fonti gratis:** FD `HY/AY/HC/AC` → `fd_side_rates.csv`; FBref match logs opzionale.
- **Marcatori:** Understat player xG + FBref; lineup FotMob su top picks.
- Cluster ML, XGB O/U+AH, conformal, residual/pesi WF, history ricca, online_learn, ritreno GHA, …
- **No-bet ragionevole:** conformal width 0.70 non è più veto; O/U–AH solo sul mercato del pick; bin OOF protetti; `online_p_factor` 1X2 n≥80, floor 0.96.
- Warning soccerdata (stagione `2021`, concat pandas, Mondiali vs club) e sklearn `y_prob` non somma a 1: stagioni `2526`, FBref solo Big 5, p rinormalizzate.
- **UI copia voto:** su singola / calendario / mercati, testo + scheda grafica in clipboard o download PNG/txt.
- **Solo quote fermo a 0%:** progresso in-process + `python -u` sugli CLI figli.
- **Score pro:** pesi fonti, Unified+Confidence+Risk 0–100, Priorità calendario, override meteo/assenze, Bet Type Recommender (`pro_scores.py`).
- **GHA Asian timeout:** `TimeoutError` su un giorno → skip (retry HTTP); `notify_cloud` non crasha se Asian è lento.
- **Settle secondari:** cards/corners FD + scorer FotMob in `history.settle_pending`.
- **Phasing-out backfill:** fit solo live quando `n_rich_live ≥ 150`.
- **GHA pre-match:** workflow `odds-prefresh.yml` (2×/giorno `--odds-update`).
- **Connettori:** `http_client` UA rotation; Betfair cache stale su errori CI.

---

## Ordine consigliato (verso 10/10)

```text
1. Live ricche 51→80+     (--odds-update pre-match quotidiano)
2. Settle cards/corners/scorer   (history.py)
3. Connettori più stabili  (Betfair/Asian/FotMob)
4. Phasing-out backfill    (quando live ≥ 150)
5. Paper Kelly su solo live ricche
```

---

*Item chiuso → spostarlo in “Chiusi di recente” e aggiornare la checklist in `PROJECT_BRIEF.md`.*
