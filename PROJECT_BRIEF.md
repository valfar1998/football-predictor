# Football Predictor — Project Brief

Sintesi aggiornata del progetto **football-predictor**  
Repo: [github.com/valfar1998/football-predictor](https://github.com/valfar1998/football-predictor) · branch `main`  
Ultimo aggiornamento: **2026-08-22** — roadmap implementativa chiusa; apprendimento trainable-only; snapshot 830 settled / 151 trainable.

Scopo di questo file: dare a un altro modello / analista contesto sufficiente per suggerire miglioramenti **senza** dover leggere tutto il codice.

**Lingua UI/note:** italiano.  
**Stack:** Python 3.10+, Streamlit, pandas, scikit-learn, XGBoost, soccerdata, statsbombpy, mplsoccer.

**Convenzione docs:** dopo ogni modifica di codice aggiornare **questo file** e `TECH_ROADMAP.md`.

---

## 1. Cos’è

App di **analisi e value betting** sul calcio (pre-match), non un bookmaker né un tipster automatico.

### Pipeline tipica

1. Scarica risultati storici + calendari + quote  
2. Costruisce feature (forma, xG rolling, Elo, casa/trasferta, `mkt_p_*`, `days_into_season`, …)  
3. Allena ML 1X2 (globale + **cluster**) + **XGB O/U 2.5 / AH 0** + conformal OOF + temperature  
4. Simula Monte Carlo → probabilità multi-mercato (+ bootstrap IC; conformal 1X2/O/U/AH)  
5. Confronta con quote (Asian, Pinnacle/Odds API, Betfair, football-data) → EV / Kelly / voto  
6. Arricchisce **quadro** + validazione + accordo dinamico + residual (gate produzione)  
7. **Archive ricco** pre-match → **settle** post-match → **`online_learn`** (calibrazione leggera)  
8. UI: calendario (CSV/JSON), tutti i mercati, singola, valutazione / paper / apprendimento  

### Principio di design (non negoziabile)

| Può entrare | Non deve entrare (senza backtest) |
|-------------|-----------------------------------|
| Feature storiche, λ, ML, MC | FotMob live / matchDetails nell’EV |
| Quote sharp/Asian per EV/Kelly | Sportly-sim / data_signal in `p_cons` grezzo |
| Quadro, voto, no_bet, Δ validazione | Gonfiaggio voto senza quote reali |
| Residual/conformal/online bins su OOF/SQLite | Soft Kelly solo se residual **produzione** |
| Cards/corners/scorer come mercati esplorativi | Corner/scorer “forti” senza match-log / lineup |

Molte fonti arricchiscono *quadro* e *voto*; **EV / Kelly** passano da probabilità calibrate (bin + `online_p_factor`), non da contesto live fragile.

---

## 1.0 È un “algoritmo”?

**In senso stretto, no.** Non esiste un solo algoritmo che mappa `(partita, quote) → pick`.  
Esiste un **sistema ibrido decisionale** — una pipeline di modelli, simulazioni, calibrazione e regole — che produce per ogni partita probabilità, edge, filtri e un voto sintetico.

| Componente | Tipo | Cosa fa |
|------------|------|---------|
| **FeatureEngineer + XGBoost** | ML supervisionato | Stima p(1/X/2), p(O/U 2.5), p(AH 0) da ~63k partite storiche |
| **Poisson / Dixon–Coles** | Modello statistico | λ gol attesi da xG rolling, meteo, forma |
| **Monte Carlo** | Simulazione stocastica | 4000 scoreline → probabilità su 15+ mercati |
| **Conformal + temperature** | Calibrazione statistica | Intervalli/set di confidenza su OOF (~33k predizioni) |
| **Reliability bins + `online_p_factor`** | Calibrazione post-hoc | Corregge over/under-confidence vs risultati reali |
| **Residual EV (Ridge)** | ML leggero online | Stima `(hit − p)` da EV, voto, accordo, steam → aggiusta edge |
| **Pesi `data_signal`** | Ottimizzazione WF | Affina peso fattori tattici nel **quadro** (non nell’EV grezzo) |
| **`advise` + `no_bet`** | Regole + euristica | EV/Kelly, veto steam/conformal/accordo, voto 1–10 |
| **`pro_scores`** | Scoring composito | Unified / Confidence / Risk 0–100, priorità calendario |

In pratica:

- **Probabilità “core”** → ML + Poisson + MC, poi calibrate.  
- **Value (EV/Kelly)** → solo su probabilità calibrate × quote verificabili.  
- **Quadro e voto** → fonti tattiche, tipster, storico squadra, override meteo/assenze — **parallelo** all’EV, non lo sostituisce.  
- **Apprendimento continuo** → aggiorna calibrazione/residual/pesi da partite chiuse **riche**, senza ri-allenare XGB ogni giorno.

Quindi è corretto dire **“il predictor”** o **“il sistema”**; “l’algoritmo” va bene solo se si intende l’intera pipeline, non un unico modello.

---

## 1.1 Come funziona il sistema (end-to-end)

### Fase offline (settimanale o `--train`)

```text
football-data CSV  ──►  FeatureEngineer  ──►  ~63k righe feature
                              │
                              ├── XGBoost 1X2 (globale + 8 cluster leghe)
                              ├── XGBoost O/U 2.5 + AH 0  (market_models)
                              ├── Walk-forward OOF  ──►  conformal.json
                              ├── Temperature scaling  ──►  calibration.json
                              └── best_model.joblib + market_models.joblib
```

Il modello impara da **risultati passati** e quote storiche (`mkt_p_*`). Non “vede” il futuro: le predizioni live usano feature calcolate **solo con dati disponibili prima del kickoff**.

### Fase live (ogni refresh calendario)

Per ogni fixture **coperta** (entrambe le squadre nello storico train):

```text
1. MatchPredictor
      ├─ routing: lega → cluster ML → fallback globale
      ├─ p(1/X/2) da XGB + temperature + bin reliability
      ├─ p(O/U 2.5), p(AH 0) da market_models (~55%) + MC (~45%)
      └─ λ gol (Understat → FotMob → FBref) + meteo

2. MonteCarloSimulator (4000 sim)
      └─ distribuzione scoreline → BTTS, multigol, exact, cards/corners proxy, …

3. value.enrich_value
      ├─ de-vig quote book / Pinnacle / Asian
      ├─ EV cons vs book, EV sharp vs Pinnacle
      ├─ Kelly ¼ (cap) su p calibrata
      └─ CLV vs chiusura se disponibile

4. advise (per ogni mercato candidato)
      ├─ accordo dinamico ML ↔ MC ↔ mercato
      ├─ conformal: pick fuori dal set → no_bet
      ├─ residual_ev: se produzione, adj_ev + primary no_bet
      ├─ no_bet: EV sotto soglia, steam contrario, accordo spezzato, …
      └─ voto 1–10 + score_unified (pro_scores)

5. quadro + validation + data_signal
      └─ fattori tattici (FBref, meteo, Elo, …) → Δ voto, NON moltiplicano p_cons

6. archive_upcoming  ──►  SQLite (snapshot pre-match)
```

Partite **N/D** (squadra fuori training): solo quadro/validazione, voto max ~3, **nessun pick EV**.

### Fase post-match (apprendimento leggero)

```text
risultato fd  ──►  settle_pending (hit 0/1)
                        │
                        ▼
              learn_from_settled()
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  reliability bins   online_p_factor   min_ev_play
        │               │               │
        └───────────────┼───────────────┘
                        ▼
              residual_ev (Ridge WF)
                        │
                        ▼
              data_signal_weights (grid WF)
```

**Importante:** il fit usa solo righe **trainable** (vedi §1.4). Le ~679 live vecchie senza quota/EV/fattori **non entrano** nel fit.

### Cosa decide se “giocare” o no

Ordine concettuale (semplificato):

1. Esiste un pick con **quota reale** (`odds_real` o equivalente)?  
2. **EV** ≥ `min_ev_play` (adattivo da ROI recente)?  
3. Pick dentro il **set conformal** del mercato?  
4. **Accordo** ML/MC/mercato sufficiente?  
5. **Residual** non blocca (primary no_bet)?  
6. Nessun **steam Asian** fortemente contrario?

Se passa → pick con Kelly; altrimenti → `no_bet` ma può restare un **voto analitico** alto (quadro forte senza edge misurabile).

### Tre “livelli” di output (non confonderli)

| Livello | Domanda | Output |
|---------|---------|--------|
| **Probabilità** | Quanto è probabile l’esito? | p ML/MC calibrata per mercato |
| **Value** | C’è edge vs il book? | EV, Kelly, CLV — solo con quote reali |
| **Voto / quadro** | Quanto è interessante analiticamente? | 1–10, Unified/Conf/Risk, fattori tattici |

Una partita può avere **voto alto** e **no_bet** (contesto interessante, quote senza value).  
Una partita può avere **EV positivo** e **voto basso** (edge sottile, disaccordo modelli).

---

## 1.2 Stato roadmap (2026-08-22)

### Roadmap implementativa (codice) — ✅ terminata

Tutti i gate tecnici previsti dalla roadmap sono **sbloccati e operativi**:

| Gate | Target | Valore attuale | Stato |
|------|--------|----------------|--------|
| Settled totali (voto storico) | ≥ 30 | **830** | ✅ |
| Righe **trainable** | ≥ 80 | **151** | ✅ |
| Residual EV WF-RMSE | ≤ 0,55 | **≈ 0,53** | ✅ produzione |
| `online_p_factor` | live ricche ≥ 30 | **30** (factor 1.005) | ✅ |
| Pesi `data_signal` | fit operativo | **655** righe replicate | ✅ |
| Apprendimento senza junk | escludere live incomplete | **679** escluse | ✅ |

Dettaglio operativo aggiornato: `TECH_ROADMAP.md`.

### Roadmap operativa (dati live) — 🟡 in corso

Non richiede nuovo codice; richiede **routine quotidiana**:

| Obiettivo | Ora | Target | Perché |
|-----------|-----|--------|--------|
| Live ricche pre-match | **51** | 80+ | Paper Kelly e ROI @ quote reali |
| `quota_pick` live | **51** | crescere | Backfill synthetic non ha quote pre-match reali |

**Routine consigliata:**

```text
Solo quote (pre-match)  →  settle automatico  →  Apprendi da partite chiuse
```

Il **backfill synthetic (100 righe)** ha già fatto da bootstrap per bins/residual/pesi.  
Da ora il sistema **impara soprattutto** da: live ricche (×5 nel fit) + backfill (×4), ignorando lo storico live incompleto.

**Conclusione:** il progetto è **pronto all’uso**; la validazione economica live (paper ROI credibile) migliora man mano che crescono le archiviazioni pre-match su Top leghe.

---

## 1.3 Da dove arrivano i dati (per fase)

### Fase A — Allenamento (`--train`, GHA settimanale)

| Fonte | Cosa scarica | Dove finisce |
|--------|----------------|--------------|
| **football-data.co.uk** | CSV stagioni: risultati, quote 1X2 open/close, cartellini HY/AY, corner HC/AC | `data/raw/fd/` → `matches.csv` |
| **football-data.org** (token) | Coppe UEFA / internazionali | `data/raw/org/` |
| **OpenLigaDB, TheSportsDB, API-Football** | Calendari extra, venue | fixtures mondiali |
| **FBref** (soccerdata, Big 5) | Poss, tiri, xG squadra, player | `fbref_team_context.csv` |
| **Understat** | xG squadra + player per marcatori | `understat_*_context.csv` |
| **StatsBomb / Sofascore / FotMob** | Open data, classifiche, match recenti | `*_team_context.csv`, `fotmob_matches.json` |
| **ClubElo** | Elo pubblico (cache 72h) | `data/raw/clubelo.csv` |
| **side_rates** (interno) | Medie HY/HC ultime partite da CSV fd | `fd_side_rates.csv` |

Il **FeatureEngineer** trasforma `matches.csv` in ~63k righe feature (forma 5 partite, xG rolling, Elo, `mkt_p_*` da quote storiche, giorni in stagione, …).  
**ModelTrainer** allena XGBoost 1X2 globale + **8 cluster** leghe; **market_models** O/U 2.5 e AH 0; **conformal** su OOF (~33k predizioni out-of-fold).

### Fase B — Calendario live (`build_upcoming`, `--update`, Solo quote)

| Fonte | Uso nel pronostico |
|--------|---------------------|
| **football-data.co.uk** fixtures | Elenco partite + quote medie book |
| **AsianBetSoccer** | Quote Asian, steam linee, spread score |
| **The Odds API (Pinnacle)** | Quote sharp (cache ~20h, 1 call/giorno) |
| **Betfair** | Exchange (soft-fail se 403; cache) |
| **Understat / FotMob xG rolling** | λ gol in `MatchPredictor` + Poisson |
| **FBref / StatsBomb / Sofascore** | Contesto tattico → **quadro**, non EV grezzo |
| **Open-Meteo + geocode** | Meteo kickoff → piccola correzione λ |
| **Tipster** (Forebet, PredictZ, Vitibet) | Consenso esterno → quadro/voto |
| **WhoScored** (opz., lento) | Assenze → override voto |

Per ogni partita **coperta dal modello**: `MatchPredictor` → `MonteCarloSimulator` (4000 sim) → `advise` (EV, Kelly, voto, no_bet, residual).  
Partite **N/D** (squadra assente dallo storico train): solo quadro/validazione, voto max ~3, niente pick EV.

### Fase C — Cosa **non** entra nell’EV

- Dettagli live FotMob / matchDetails non in cache pre-match  
- Sportly-sim e `data_signal` **non** moltiplicano `p_cons` grezza  
- Gonfiaggio voto senza `odds_real`  
- Cards/corners/scorer: mercati esplorativi MC, **senza settle** in SQLite  

---

## 1.4 Come cresce e si affina (partite chiuse)

Il ciclo di apprendimento **leggero** (non ri-allena XGB ogni giorno) funziona così:

```text
  PRE-MATCH                          POST-MATCH
  ─────────                          ──────────
  build_upcoming / Solo quote
         │
         ▼
  archive_upcoming()  ──►  SQLite our_history.sqlite
  (pick, EV, quota_pick,             (campi ricchi:
   data_factors, agree, …)            ev_sharp, cluster, …)
         │                                    │
         │         partita finita             │
         └──────────────────────────────────► settle_pending()
                                              (hit 0/1 da risultati fd)
                                                       │
                                                       ▼
                                              learn_from_settled()
                                              (UI: Apprendi da partite chiuse)
```

### Cosa viene salvato pre-match (`archive_upcoming`)

Ogni riga del calendario (anche N/D) può finire in SQLite. Per l’apprendimento **utile** servono soprattutto le righe **coperte** con:

- `quota_pick` — quota reale del pick (preferenza: `odds_real` → odds mercato → fair)  
- `ev_cons` / `ev_sharp` — edge vs book / vs sharp  
- `data_factors` — JSON fattori analisi dati (FBref, meteo, …)  
- `agree_share` — accordo ML / MC / mercato  
- `probability`, `score_unified`, `model_cluster`, `pick_group`, …  

**Importante:** se archivi solo *dopo* il match o senza quote, la riga settled **non** alimenta residual/paper in modo credibile (volume alto ≠ qualità).

### Cosa fa `settle_pending`

Confronta date/squadre con risultati in `matches.csv` / fd e imposta `home_goals`, `away_goals`, `hit` (1 se pick vincente sul mercato scelto).  
Supporta 1X2, DC, O/U, BTTS, multigol, parity, exact, AH0 — **non** cards/corners/scorer.

### Cosa fa `learn_from_settled` (online learning)

| Step | Effetto | Gate attuale (aggressive) |
|------|---------|---------------------------|
| Reliability bins online | Blend fino al **68%** con bin OOF se ogni bin ≥ ~12–30 settled | ✅ su 655 righe replicate |
| `online_p_factor` | Correzione soft probabilità 1X2 da bias recente | **≥ 30 live ricche** 1X2 |
| `min_ev_play` adattivo | Alza/abbassa soglia EV da ROI trainable @ quote | trainable con `quota_pick` |
| **Residual EV** | Ridge walk-forward su (hit − p) vs EV, voto, accordo, steam | **trainable ≥ 80** → produzione |
| **Pesi data_signal** | Grid + walk-forward Brier/ROI su fattori quadro | trainable con `data_factors` |

**Non fa:** ritrain XGB/cluster (quello è `--train` o GHA `weekly-train.yml`).

### Policy apprendimento: solo righe **trainable** (`learn_policy.py`)

Una riga è **trainable** (e entra nel fit) solo se, oltre a `hit` settled, ha **tutti**:

- `quota_pick`
- `ev_cons` o `ev_sharp`
- `data_factors` (JSON)
- `agree_share`

| Tipo riga | Esempio | Nel fit? | Peso replicate |
|-----------|---------|----------|----------------|
| **Live ricca** | archivio pre-match con quote+EV+fattori | ✅ sì | **×5** |
| **Backfill synthetic** | `--backfill-history`, `synthetic_backfill=1` | ✅ sì | **×4** |
| **Live vecchia incompleta** | settled senza quota o fattori (~679) | ❌ no | — |
| **N/D o solo quadro** | partita fuori modello | ❌ no | — |

Implementazione: `modules/advisor/learn_policy.py` — `is_rich()`, `is_trainable()`, `replicate_for_fit()`.

Con `aggressive_learn: true` (default in `calibration.json`):

- Fit bins / residual / pesi su **151 trainable** → **655 righe replicate** (51×5 + 100×4)
- `online_p_factor`: priorità chunk **live ricche** (≥30); fallback pool trainable (≥60)
- ROI per `min_ev_play`: prima live ricche; poi tutto il trainable
- Blend bin online più aggressivo (68% vs 40% conservativo)

```powershell
python main.py --backfill-history --backfill-max 120   # bootstrap una tantum (già fatto: 100)
python main.py --odds-update                            # Solo quote pre-match (routine)
# dopo settle → learn_from_settled() automatico o bottone Apprendi in UI
```

### Snapshot storico (2026-08-22, post-backfill + learn)

| Metrica | Valore |
|---------|--------|
| Righe in SQLite | **3 952** |
| Settled totali | **830** |
| **Trainable** (fit apprendimento) | **151** ✅ |
| — live ricche | **51** |
| — backfill synthetic | **100** |
| Skipped (live incomplete) | **679** |
| Residual EV | **produzione** — n=620 fit, WF-RMSE ≈ **0,53** |
| `online_p_factor` | **1,005** (30 live ricche 1X2) |
| Pesi data_signal | Brier ≈ **0,25** su 655 replicate |

Lo storico entra nel **voto unificato** (peso ~15%) quando ≥ 30 settled globali e ≥ 6 per squadra (`history.py`).  
Le righe incomplete restano in SQLite per il voto, ma **non inquinano** calibrazione/residual/pesi.
### Resilienza rete e bootstrap (2026-08-22)

| Strategia | Implementazione |
|-----------|-----------------|
| **curl_cffi** | `modules/data_update/http_client.py` — FotMob usa TLS fingerprint browser; fallback urllib se assente |
| **Degradazione controllata** | `archive_upcoming`: try/except già in pipeline; flag `context_partial` se mancano fattori/quote; core EV da fd + Odds API |
| **Cache 72h** | `cache_policy.py` + skip in `download_all` e refresh leggero (FBref, Understat, FotMob, …) |
| **Backfill storico** | `python main.py --backfill-history [--backfill-max 120]` — synthetic da `matches.csv` (quote close), `synthetic_backfill=1` |

Il backfill **non sostituisce** archivi live ricchi; serve a sbloccare gate (`online_p_factor`, storico ricco) più in fretta. Probabilità con modello attuale su passato = leakage leggero accettabile solo per bootstrap calibrazione.

---

## 2. Struttura cartelle

```
football-predictor/
├── PROJECT_BRIEF.md
├── TECH_ROADMAP.md
├── app.py / main.py
├── data/
│   ├── raw/          # token, cache odds (gitignored secrets)
│   ├── processed/    # features, upcoming, context, our_history.sqlite
│   └── models/       # best_model.joblib, market_models.joblib,
│                     # conformal.json, calibration.json, oof_*.joblib,
│                     # residual_ev.json, data_signal_weights.json, …
├── modules/
│   ├── data_update/     # download, odds, FotMob(+lineup), weather, history, side_rates (FD)
│   ├── feature_engineering/
│   ├── model_training/  # train.py, league_clusters.py, market_models.py
│   ├── predictor/       # predict.py (+ market_ml), poisson.py
│   ├── montecarlo/      # simulate.py + extras.py (FD→FBref→proxy)
│   ├── advisor/
│   │   ├── advise.py, value.py, staking.py, quadro.py, validation.py
│   │   ├── data_signal.py, data_signal_weights.py, scorers.py
│   │   ├── agreement.py, residual_ev.py, paper_stats.py
│   │   ├── learn_policy.py   # filtro trainable, replicate live×5 / backfill×4
│   │   ├── pro_scores.py   # pesi fonti, Score/Conf/Risk 0–100, priorità, override, bet-rec
│   │   └── online_learn.py
│   ├── calibration/     # calibrate.py, conformal.py, config.py
│   ├── sportly_sim/, tipsters/, notify/, visualization/
├── scripts/             # notify_cloud.py, cloud_bootstrap.py, VBS refresh
└── .github/workflows/   # telegram-asian-alerts, cloud-train, weekly-train
```

**Eliminati (non ripristinare):** `sportly-master/`, `Public-FotMob-API-main/`.

---

## 3. Modello predittivo (nucleo)

| Pezzo | Ruolo | Note |
|--------|--------|------|
| `FeatureEngineer` | Forma, xG/xGA, WR, rest, Elo, `mkt_p_*`, `days_into_season` | Serve `--train` |
| `league_clusters` | Keyword + **similarità statistica** (`league_stat_profiles.json`) | Fallback `global` |
| `ModelTrainer` | XGB globale + XGB per cluster (`n≥800`) | Bundle tipico: **8 cluster** + globale |
| `market_models` | XGB binari **O/U 2.5** + **AH 0** + temperature | `market_models.joblib`; CLI `--train-markets` |
| `poisson` / Dixon–Coles | λ | Understat → FotMob rolling → FBref; meteo |
| `MatchPredictor` | `_model_for(league)` + `market_ml` + conformal 1X2 | T: lega → cluster → globale |
| `MonteCarloSimulator` | Scoreline → mercati estesi | `ah_home_0`; + `extras` cards/corners |
| `conformal.py` | 1X2 + O/U + AH; preferisce OOF XGB | Fallback Poisson λ; fix chiave `over_2.5` |
| Calibrazione | T globale/lega/**cluster**, reliability bins, `online_p_factor` | Anche da settle |

**Output predizione tipici:** `model_probabilities`, `market_ml` (`p_over_25`, `p_ah0_home`), `expected_goals`, `features`, `montecarlo`, `model_cluster`, `conformal_intervals`.

**Blend O/U 2.5 e AH 0 in `advise`:** ~55% XGB + ~45% MC.

---

## 4. Mercati supportati (`advise` + MC)

| Gruppo UI | Contenuto | Qualità |
|-----------|-----------|---------|
| `1x2` | 1 / X / 2 | Core (ML+MC) |
| `dc` | 1X, 12, X2, DNB | Core |
| `ah` | Asian Handicap 0 casa/ospite | Core (**XGB AH0** + MC) |
| `ou` | Over/Under 0.5–4.5 | Core (**O2.5: XGB O/U** + MC) |
| `btts` | Gol / No gol (GG/NG) | Core |
| `multigol` | 0-1, 1-2, 2-3, 3-4, 1-3, 2-4, 0-2, 3+, 4+ | Core (dist. gol MC) |
| `parity` | Pari / Dispari gol totali | Core |
| `exact` | Top 6 risultati esatti | Core (prob. basse) |
| `team` | Gol squadra O/U + vince a 0 | Core |
| `combo` | 1X2+O/U, 1X2+BTTS, Gol+O2.5, … | Core |
| `cards` | Cartellini O/U 2.5–5.5 | **FD rates** (HY/AY) → FBref match logs → season → proxy |
| `corners` | Corner O/U 8.5–11.5 | **FD rates** (HC/AC) → FBref match logs → crosses → proxy |
| `scorer` | Anytime / first | Understat+FBref xG; **boost lineup FotMob** se disponibile |

Artefatti: `market_models.joblib`, `fd_side_rates.csv`, `understat_player_xg.csv`, `fbref_match_side_rates.csv` (opz.).

Quote: se assenti → probabilità sì, EV/Kelly con quota **stimata**; consiglio “gioca” forte legato a `odds_real` dove possibile.  
`quota_pick` in archive: preferisce `odds_real` → `odds` → `fair_odds`.

Settle (`history._hit_for_pick`): O/U, BTTS, multigol, parity, exact, DC, **AH0** — non cards/corners/scorer.

---

## 5. Advisor (value + voto + filtri)

### 5.1 Value / EV (`value.py`)
De-vig, EV cons vs sharp, Kelly ¼; `calibrated_prob` = p × reliability bin (n≥30) × `online_p_factor` **solo 1X2** (cap ±6%, da bias live ricche se n≥30). Soft Kelly se residual in produzione. IC 1X2 fragile → Kelly ×0.70, non no_bet.

### 5.2 Filtri `no_bet`
**Veto:** EV/sharp sotto `min_ev_play`, quota ipotetica, steam contrario, **pick fuori dal set conformal del suo mercato**, 1X2 con p_cons < 32%, accordo spezzato, residual primary (solo se produzione).  
**Non veto:** IC largo, set 1X2 a 3 esiti, O/U/AH conformal su un pick di un altro mercato. Quelli vanno su voto/Kelly.

### 5.3–5.4 Voto e quadro
Gambe value/kelly/asian/workflow/history/combos + Δ; quadro da fonti ortogonali **non** mischiate nell’EV.

### 5.5 Analisi dati + pesi WF
`data_signal` + pesi walk-forward Brier/ROI (global / cluster / market) → solo quadro/Δ voto.

### 5.6 Residual EV
Ridge WF + cluster; gate **n≥80** settled con EV; `adj_ev`, soft Kelly, primary no_bet.

### 5.7 Paper trading
Flat, ROI @ quota, Kelly equity, max DD, Sharpe, WF ROI, breakdown.

### 5.8 Online learning (`online_learn.py` + `learn_policy.py`)
Dopo settle / bottone **Apprendi da partite chiuse**: fit **solo su righe trainable** (live ricche + backfill synthetic); live incomplete escluse. Aggiorna reliability bins (blend fino 68% se aggressive), `online_p_factor`, `min_ev_play`, residual fit, pesi data_signal.  
**Non** ritrena XGB a ogni partita (`--train` / GHA settimanale).  
**Non** sovrascrivere i bin OOF (33k) con pochi settled — si fa **blend**, non replace totale.

### Cosa cambia le proposte

| Cambio | Impatto |
|--------|---------|
| Cluster ML + T + market XGB | p 1X2 / O2.5 / AH0 → EV |
| FotMob xG / meteo | λ → MC → mercati gol |
| Bin + `online_p_factor` | p calibrata → EV/Kelly/voto |
| Accordo / conformal / residual | no_bet, Kelly, voto |
| Nuovi mercati / scorer | filtri “Tutti i mercati” |

---

## 6. History ricca (`history.py`)

SQLite `our_history.sqlite` — campi ricchi:

`quota_pick`, `ev_cons`, **`ev_sharp`**, `agree_share`, `data_edge`, `move_rank`, `residual`, `adj_ev`, `data_factors` (JSON), `no_bet_reasons`, `pick_group`, `model_cluster`, voto/`score`

`archive_upcoming` pre-match → `settle_pending` → `learn_from_settled`.  
Senza refresh pre-match con quote, residual/paper Kelly restano poveri (volume ≠ qualità campi).

---

## 7. Fonti dati

| Fonte | Uso |
|--------|-----|
| football-data.co.uk / .org | Storico, fixtures, quote avg; **HY/AY/HC/AC** → `fd_side_rates.csv` |
| TheSportsDB, API-Football, OpenLigaDB | Coppe / mondo / venues |
| The Odds API (Pinnacle), Betfair, AsianBetSoccer | Quote sharp / exchange / Asian |
| ClubElo, FBref **Big 5** (team + player + match logs opz.), Understat (team + **player xG**), StatsBomb, Sofascore | Contesto |
| WhoScored | Assenze (lento) |
| Open-Meteo | Meteo + geocode |
| FotMob `/api/data` | Classifiche, match, rolling xG, details + **lineup XI** top picks (cache 6h) |

**Betfair:** soft-fail (`betfair_soft_fail` in `notify_cloud` / pipeline); GHA spesso 403 → continua con Pinnacle/Asian.

**AsianBetSoccer (GHA):** timeout/404 su un giorno → skip + retry HTTP; `notify_cloud` non fallisce più se Asian è lento/vuoto (tiene la cache precedente).

**Non vendorizzare:** Sportly SDK, Public-FotMob Django, bypass WAF aggressivi.

---

## 8. UI Streamlit

| Tab | Contenuto |
|-----|-----------|
| Calendario | Filtri gruppi (anche `ah` / `scorer`), voto, quadro, CSV + JSON |
| Tutti i mercati | Flat multi-gruppo |
| Singola | Predict on demand |
| Valutazione | Calibrazione, disaccordi + timeline, paper, apprendi, pesi, residual |

Sidebar: fonti contesto + **Geocode**; caption **Modelli cluster** + **O/U/AH**; Telegram/GHA.  
Bottoni lunghi: barra **% + log live** (`modules/progress_report.py`) e echo su terminale Streamlit.  
**Quote leggere:** Betfair / Pinnacle / Asian / tipster → `refresh_upcoming_odds` (stesse p ML/MC, ricalcola solo EV/Kelly/voto).  
**Solo quote / `--odds-update`:** fixtures fd + Asian + Pinnacle/Betfair (se cache scaduta) + `build_upcoming` con riuso ML/MC. Non scarica mondiale, coppe extra, tipster, FBref/FotMob. Monte Carlo solo sulle partite nuove. Coppe/mondo: bottoni dedicati. (**nota:** `--update` / Aggiorna dati + modello riscarica tutto e riallena → ricalcolo MC pieno lungo).  
**Modello da cloud:** `python main.py --pull-model` (o bottone UI) scarica l’artefatto Actions (`weekly-model-*` / `best-model-*`, include `features.csv`); poi Solo quote.  
**Progresso Aggiorna dati + modello:** righe `calendario N/M` mappano 95→99% (`progress_report.py`).  
**notify-refresh:** path leggero se esiste già il calendario.  
Filtro date: default **da oggi**.

---

## 9. CLI

```powershell
cd C:\Users\valba\Desktop\corsi\football-predictor
.\.venv\Scripts\Activate.ps1
python main.py --train              # 1X2 + cluster + market O/U-AH + conformal
python main.py --train-markets      # solo O/U 2.5 + AH 0 (veloce)
python main.py --odds-update        # Solo quote: fd + quote + calendario (riuso ML/MC)
python main.py --asian-odds         # Asian + tipster + refresh value leggero
python main.py --backfill-history   # synthetic settled da matches.csv (gate 80)
python main.py --backfill-history --backfill-max 150
python main.py --predict "Inter" "Milan"
streamlit run app.py
```

---

## 10. Limiti noti

1. API non ufficiali (FotMob/FBref/Sofascore) fragili.  
2. Residual produzione: trainable ≥ 80; paper Kelly credibile solo su **live ricche** con `quota_pick` (51 oggi).  
3. Backfill synthetic: bootstrap calibrazione ok, **non** sostituisce quote pre-match reali.  
4. Cards/corners: FD rates; niente settle hit. Marcatori: quote book rare.  
5. Betfair CI 403; soft-fail gestito.  
6. Conformal O/U–AH: OOF XGB preferito, altrimenti Poisson λ.  

---

## 11. Stato roadmap (checklist)

### Implementazione (codice) — completata ✅

| Area | Stato |
|------|--------|
| ML multi-cluster + routing similarità | ✅ |
| XGB O/U 2.5 + AH 0 + conformal OOF | ✅ |
| Residual EV produzione (WF-RMSE ≈ 0,53) | ✅ |
| Pesi data_signal WF su trainable | ✅ |
| Online learning trainable-only (`learn_policy`) | ✅ |
| Backfill synthetic bootstrap (100 righe) | ✅ |
| Aggressive learn (live×5, backfill×4) | ✅ |
| `online_p_factor` da live ricche (n=30) | ✅ |
| Mercati estesi, UI pro_scores, GHA settimanale | ✅ |

### Operatività (dati live) — in corso 🟡

| Area | Stato | Nota |
|------|--------|------|
| Trainable totali | **151 / 80** ✅ | 100 backfill + 51 live ricche |
| Live ricche pre-match | **51 / 80+** | cresce con **Solo quote** pre-KO |
| Paper Kelly @ quote reali | **parziale** | solo live con `quota_pick` |

Dettaglio gate e azioni: `TECH_ROADMAP.md`.

---

## 12. File da leggere per primi

1. `PROJECT_BRIEF.md` · `TECH_ROADMAP.md`  
2. `modules/advisor/advise.py` · `pro_scores.py` · `learn_policy.py` · `online_learn.py` · `agreement.py` · `value.py` · `staking.py` · `residual_ev.py` · `scorers.py` · `vote_copy.py`  
3. `modules/model_training/train.py` · `market_models.py` · `league_clusters.py`  
4. `modules/predictor/predict.py` · `montecarlo/simulate.py` · `extras.py`  
5. `modules/calibration/conformal.py` · `calibrate.py`  
6. `modules/data_update/history.py` · `upcoming.py` · `fbref_context.py`  
7. `app.py` · `main.py` · `.github/workflows/weekly-train.yml`  

---

## 13. Snapshot decisionale

> **Sistema ibrido** (non un singolo algoritmo): ML cluster 1X2 + XGB O/U/AH + Poisson + MC multi-mercato → probabilità calibrate (bin OOF + blend online trainable-only + `online_p_factor`); edge da quote sharp/Asian; quadro tattico separato dall’EV; no_bet su filtri value/conformal/accordo/residual.  
> **Apprendimento:** pre-match `archive_upcoming` → post-match `settle` → **Apprendi** aggiorna calibrazione/residual/pesi **senza** ritrenare XGB (ritrain GHA settimanale). Fit solo su **151 trainable** (51 live ricche + 100 backfill); **679 live incomplete ignorate**. Residual in produzione (WF-RMSE ≈ 0,53).  
> **Roadmap codice:** ✅ chiusa. **Roadmap dati:** far crescere live ricche (51→80+) con **Solo quote** pre-match per paper Kelly credibile.

---

*Aggiornare questo brief e `TECH_ROADMAP.md` a ogni modifica di codice (mercati, EV, filtri, API, train, CI).*
