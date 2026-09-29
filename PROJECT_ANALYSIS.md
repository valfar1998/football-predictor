# Football Predictor — Analisi Tecnica Completa

**Ruolo del documento:** audit architetturale e di sistema (Senior AI Engineer / System Architect).  
**Ambito:** codebase di produzione in `football-predictor` (escluso il dettaglio interno di `soccerdata-master/` oltre al ruolo di libreria vendored).  
**Data analisi:** 2026-09-29.

---

## 1. Panoramica dell’Architettura (System Overview)

### 1.1 Natura del sistema

Il progetto è un **pipeline ibrido di value betting pre-match**, non un semplice classificatore 1X2. Combina:

1. **Supervised ML** (XGBoost / Random Forest) per probabilità di esito e mercati binari.
2. **Modello di gol analitico** (Dixon–Coles su Poisson) per λ e probabilità 1X2 alternative.
3. **Simulazione Monte Carlo Poisson** per mercati derivati (O/U, BTTS, DC, combo, multigol, extras).
4. **Stack di value / staking** (devig, calibrazione, edge conservativo, ¼ Kelly, filtri no-bet).
5. **Apprendimento a due livelli**: retrain pesante del modello base + online learn leggero su settle (calibrazione, soglie, residual EV, pesi analitici).

La regola di design esplicita (ripetuta in `tactics.py`, quadro, tipster) è: **tattica, ClubElo display, data-signal e tipster non devono gonfiare l’EV grezzo**; influenzano voto/UI/validazione, non il core economico.

### 1.2 Flusso dati end-to-end

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ ACQUISIZIONE                                                            │
│  football-data.co.uk (CSV storici + fixtures + zip stagione)            │
│  football-data.org (coppe) · ClubElo · FBref/Understat/Sofascore/…      │
│  Quote: AsianBetSoccer · Odds API (Pinnacle/Codere) · Betfair · Kambi        │
│  Contesto live: weather (Open-Meteo), FotMob, StatsBomb, WhoScored      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STORICO TRAINING                                                        │
│  parse → DatasetLoader → matches.csv                                    │
│  FeatureEngineer → features.csv (+ feature_state.json incrementale)     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TRAINING (manuale / GHA daily+weekly)                                   │
│  OOF walk-forward XGB → best_model.joblib (RF vs XGB per log-loss)      │
│  cluster XGB per lega · market_models (O/U 2.5, AH0)                    │
│  temperature / reliability bins · conformal sets                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ INFERENZA PER PARTITA                                                   │
│  MatchPredictor: feature row + mkt_* live → proba ML                    │
│  λ da xG rolling + xG esterno + weather → Dixon–Coles blend             │
│  MonteCarloSimulator(λ, model_probs) → mercati                          │
│  advise → enrich_value → staking/Kelly → voto unificato / no_bet        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                                  │
│  upcoming_predictions.json · last_prediction.json                       │
│  Streamlit (Calendario / Analisi / Valutazione / Paper ROI)             │
│  Telegram alerts + freeze score (allineamento ROI)                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CICLO DI APPRENDIMENTO                                                  │
│  archive_upcoming → our_history.sqlite                                  │
│  settle_pending (Sofascore Big5 → FD/cups → world)                      │
│  learn_from_settled → calibration / residual_ev / data_signal_weights   │
│  (XGB base invariato fino al prossimo --train / GHA)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Orchestratori principali**

| Entry point | Ruolo |
|-------------|--------|
| `main.py` | CLI: `--update`, `--train`, `--predict`, `--odds-update`, `--ui`, `--pull-model` |
| `modules/data_update/upcoming.py` → `build_upcoming` | Batch calendario predizioni |
| `app.py` | UI Streamlit |
| `.github/workflows/cloud-train.yml` | Retrain giornaliero ~05:00 UTC |
| `.github/workflows/weekly-train.yml` | Train pesante domenicale |
| `.github/workflows/odds-prefresh.yml` | Refresh quote 4×/giorno |
| `.github/workflows/telegram-asian-alerts.yml` | Alert ogni ~30′ |
| `scripts/cloud_learn.py`, `scripts/notify_cloud.py` | Learn + notify post-cloud |

### 1.3 Persistenza e artefatti

| Layer | Path tipici | Formato |
|-------|-------------|---------|
| Raw storici | `data/raw/fd/`, `data/raw/org/`, `clubelo.csv` | CSV / zip |
| Cache quote | `data/raw/*_odds.json` | JSON |
| Feature matrix | `data/processed/features.csv` | CSV (~decine di migliaia di righe) |
| Calendario live | `data/processed/upcoming_predictions.json` | JSON |
| Storia operativa | `data/processed/our_history.sqlite` | SQLite (`matches`) |
| Modelli | `data/models/best_model.joblib`, `market_models.joblib`, OOF, conformal | joblib/JSON |
| Calibrazione online | `calibration.json`, `residual_ev.json`, `data_signal_weights.json`, `online_learn_report.json` | JSON |
| Digest umano | `data/models/learn_digest.md` | Markdown |

Non esiste un RDBMS esterno: lo storico di training è **CSV**, lo storico di giocate/settle è **SQLite**.

### 1.4 Stack tecnologico

| Categoria | Tecnologie |
|-----------|------------|
| Linguaggio | Python 3.10+ |
| Dati | pandas, numpy |
| ML | scikit-learn (RF, metriche, LabelEncoder), XGBoost (`multi:softprob` / `binary:logistic`), joblib |
| Statistica gol | Poisson i.i.d. + correzione Dixon–Coles (`modules/predictor/poisson.py`) |
| Simulazione | NumPy RNG Poisson (`MonteCarloSimulator`) |
| UI | Streamlit |
| Scraping / contesto | `soccerdata` locale (`-e ./soccerdata-master`), curl_cffi, statsbombpy, mplsoccer |
| Automazione | GitHub Actions, Telegram Bot API |
| Persistenza | filesystem + sqlite3 stdlib |

Dipendenze dichiarate in `requirements.txt`; train cloud usa `requirements-cloud.txt` (senza Streamlit/soccerdata).

---

## 2. Motore di Machine Learning e Predizione

### 2.1 Modelli impiegati e scopo

| Modello | Algoritmo | Artefatto | Scopo |
|---------|-----------|-----------|--------|
| **Best 1X2** | `XGBClassifier` (multi:softprob) e/o `RandomForestClassifier`; selezione per **log-loss** su holdout temporale | `best_model.joblib` | P(H), P(D), P(A) |
| **Cluster 1X2** | XGB per cluster di leghe (soglia ~≥800 righe) | chiave `models` nel bundle | Specializzazione geografica/lega |
| **O/U 2.5** | XGB binario + temperature scaling | `market_models.joblib` | P(over 2.5) |
| **AH 0 home** | XGB binario + temperature | stesso | P(casa copre AH 0); push/pareggio → label 0 |
| **Dixon–Coles** | Griglia Poisson 0…max_goals × τ(ρ) | infer-only | 1X2 analitico da λ |
| **Temperature / reliability** | Grid-search T; bin empirici OOF + online | `calibration.json` | Soften/sharpen probabilità |
| **Conformal** | Nonconformity da OOF | conformal JSON | Prediction set ~90% (veto in staking) |
| **Residual EV** | Ridge su meta-feature EV | `residual_ev.json` | Correzione bias edge / soft filter |
| **Online p-factor / min_ev** | Statistiche da settle | `calibration.json` | Soglie adattive |
| **Data-signal weights** | Ottimizzazione walk-forward | `data_signal_weights.json` | Pesi quadro/analisi (non EV grezzo) |

**Iperparametri tipici (train 1X2 finale):** XGB ~350 alberi, depth 5, lr 0.06; RF 400 alberi, depth 12, class_weight balanced. OOF rolling usa XGB più leggero (~220 estimators). Metriche OOF riportate: accuracy, log-loss, AUC-OVR, Brier multiclass, ECE.

**Snapshot metriche recenti** (`data/models/metrics.json`): best = **xgboost**; log-loss ≈ 1.013, accuracy ≈ 0.498, ECE ≈ 0.011 (migliore calibrazione vs RF ECE ≈ 0.029). I cluster Big5 hanno ECE più alto (0.05–0.08); cluster latam sensibilmente più deboli (accuracy ~0.34, ECE ~0.18) — segnale di **capacità ineguale tra leghe**.

### 2.2 Ensemble all’inferenza

In `MatchPredictor.predict` (`modules/predictor/predict.py`):

1. Ricostruzione feature row dallo stato squadra (minimo storico: tipicamente ≥3 match per lato).
2. Iniezione `mkt_*` da quote live se disponibili.
3. Routing modello globale vs cluster (`_model_for(league)`).
4. `predict_proba` → blend con Dixon–Coles: **ML weight 0.62**, DC **0.38**.
5. Calibrazione temperatura/league.
6. `predict_markets` → `p_over_25`, `p_ah0_home`.
7. λ via `lambdas_from_features` (vedi sotto) → input Monte Carlo.

Label storico dell’ensemble: `"xgb+dixon-coles"` (anche se il best_model può essere RF in un ciclo futuro).

### 2.3 Feature engineering

**Motore:** `FeatureEngineer` in `modules/feature_engineering/features.py`.  
Finestre rolling default **5** match; Elo interno K=18, start 1500; solo informazione **pre-kickoff** (no leakage sul risultato corrente).

**Colonne modello (`FEATURE_COLS`):**

| Categoria | Variabili |
|-----------|-----------|
| Forma | `home_form_pts`, `away_form_pts`, `home_form_gd`, `away_form_gd` |
| xG rolling (proxy da gol/storico FD) | `home_xg_avg`, `away_xg_avg`, `home_xga_avg`, `away_xga_avg`, `xg_diff`, `xga_diff` |
| Gol | `home_gf_avg`, `away_gf_avg`, `home_ga_avg`, `away_ga_avg` |
| Forza casa/trasferta | `home_home_wr`, `away_away_wr` |
| Elo interno | `home_elo`, `away_elo`, `elo_diff` |
| Calendario / congestione | `month`, `weekday`, `home_rest_days`, `away_rest_days`, `rest_diff`, `home_matches_7d`, `away_matches_7d`, `congestion_diff`, `days_into_season` |
| Mercato (close/open → probabilità implied) | `mkt_p_home`, `mkt_p_draw`, `mkt_p_away`, `mkt_overround`, `mkt_has` |

### 2.4 Contesto esterno e λ (attacco/difesa attesi)

**λ construction** (`lambda_utils.py`):

- Baseline lega: home **1.35**, away **1.15**.
- Mix rolling xG/xGA + baseline; clip xG in [0.05, 3.5].
- Blend storico vs xG esterno (Understat/FotMob/FBref): **hist 0.62 / ext 0.38**.
- Smoothing EMA verso baseline (`LAM_SMOOTH_ALPHA=0.25`).
- Weather può applicare `lambda_adj` (non è una feature tabellare del classificatore).

**Tattica** (`advisor/tactics.py`): possession, crosses, shot distance, recoveries (FBref style), fatigue/rest, assenze (WhoScored × peso xG+xA FBref). Produce `p_tactical` e contributi al voto; **EV/Kelly restano sul modello**.

**Data signal / quadro:** forma, casa/trasferta, xG, Understat, classifica, Elo ClubElo, Sofascore, StatsBomb, FotMob, tipster (Forebet/PredictZ/Vitibet) — aggregati per accordo/validazione UI.

**Extras Monte Carlo** (`montecarlo/extras.py`): tassi cartellini/corners/tiri da FD → FBref → proxy da λ; mercati laterali indipendenti, coverage settle storicamente più debole.

---

## 3. Simulazioni Monte Carlo

### 3.1 Implementazione

Classe `MonteCarloSimulator` in `modules/montecarlo/simulate.py`.

**Meccanica:**

1. `hg ~ Poisson(λ_home)`, `ag ~ Poisson(λ_away)` indipendenti (`numpy.random.default_rng`, seed default 42).
2. Aggregazione frequenze empiriche su `n` simulazioni.
3. Mercati derivati: 1X2, O/U linee (0.5…4.5), team O/U, BTTS, Double Chance, DNB, clean sheet, multigol/fasce, combo (es. `combo_1_o25`), top scorelines.
4. **Blend opzionale** con probabilità ML: default `blend=0.35` →  
   `p = (1−0.35)·p_MC + 0.35·p_ML`, poi rinormalizzazione simplex.
5. Intervalli bootstrap opzionali su 1X2 (`_mc_intervals`, ~40 resample, banda ~90%).
6. Extras: Poisson indipendenti per cards/corners/shots se forniti in `extras`.

Il motore **Sportly** (`sportly_sim/engine.py`) è una simulazione minuto-per-minuto a scopo **visualizzazione** (xG/pressione); **non** alimenta EV/Kelly.

### 3.2 Numero di iterazioni per contesto

| Contesto | `n_sims` tipico |
|----------|-----------------|
| Default classe / `predict_pipeline` | **10 000** |
| Slider Streamlit | 2 000–20 000 (default **8 000**) |
| `build_upcoming` (calendario) | **4 000** |
| `history_backfill` | **1 500** |
| Cloud fast build | **400** |

Trade-off esplicito: stabilità delle code (combo/multigol) vs latenza del batch calendario.

### 3.3 Da distribuzioni a Value Bet

```text
λ_h, λ_a
  → frequenze MC (p mercati)
  → advise(): probabilità primarie 1X2 da MC (fallback ML)
  → O/U 2.5 e AH0: blend 55% market-XGB + 45% MC
  → EV grezzo = p × odds − 1
  → enrich_value (value.py):
       de-vig → p_market
       reliability bins × online_p_factor → p_cal
       haircut divergenza ML–MC / bin sottili → p_cons
       edge_pp = p_cons − p_market
       ev_cons, ev_sharp (Pinnacle/Asian), ev_adj (realization × steam)
  → staking: min edge 2.5%, min p 1X2 32%, ¼ Kelly cap 2% bankroll
  → no_bet_reasons (steam, disaccordo fonti, conformal veto, quote non reali)
  → voto 1–10 / score_unified → output consigliato
```

Quote **stimata**/ipotetiche non ricevono voto value. Fonti reali tipiche: book, asianbetsoccer, pinnacle, codere_it, betfair, kambi_unibet.

**Ottimizzazione runtime:** se una fixture in calendario ha già ML/MC e il modello non è cambiato, il refresh quote **riesegue solo `advise`** (niente nuove 10k simulazioni).

---

## 4. Meccanismo di Retraining e Storico (ciclo di apprendimento)

### 4.1 Due loop distinti

Il sistema **non** confonde “apprendere dagli esiti” con “riallenare XGB”.

#### A) Retrain del modello base (pesante)

| Trigger | Dove | Cosa aggiorna |
|---------|------|----------------|
| Daily ~05:00 UTC | `cloud-train.yml` → `cloud_bootstrap.py` | Feature + XGB/RF + market models |
| Weekly domenica 04:00 UTC | `weekly-train.yml` | Train pesante completo |
| Manuale | `python main.py --train` / UI “Aggiorna dati + modello” | Idem |
| Pull | `--pull-model` | Scarica artefatto GHA senza train locale |

Pipeline tipica: `download_all` → `FeatureEngineer` → `ModelTrainer.train` (rolling OOF + holdout + cluster) → `train_market_models` → calibrazione/conformal → `best_model.joblib`.

#### B) Online learn (leggero, post-settle)

Catena:

```text
archive_upcoming() → settle_pending(learn=True) → learn_from_settled()
```

**`learn_from_settled`** (`online_learn.py`) aggiorna:

1. **Reliability bins** 1X2 (blend con bin OOF, cap aggressivo/conservativo).
2. **`online_p_factor`** da errore medio p−hit.
3. **`min_ev_play`** da ROI/CLV recente (solo voto unificato ≥8).
4. **`fit_residual_ev()`** (Ridge; gate RMSE walk-forward tipicamente ≤ ~0.55, `MIN_FIT≈80`).
5. **`optimize_weights()`** data-signal (walk-forward Brier/ROI).

Gate: ≥ **25** righe trainable (salvo `force=True`). Policy in `learn_policy.py`:

- Solo righe **ricche** (quota + EV + data_factors + agreement).
- Live ricche replicate ×5 (×6 se ≥80); backfill synthetic con decay (×4/×2/×1).
- Phase-out backfill a **≥150** live rich.
- Recency half-life **90 giorni**.
- Storico live incompleto **escluso** dal fit.

Digest recente (2026-09-29): 622 settled, 560 trainable (440 live + 120 backfill); residual RMSE ≈ 0.48; ROI voto≥8 ancora **non calcolabile** (campione insufficiente) — coerente con i livelli obiettivo n=15/30/40 in `ROI_SAMPLE_LEVELS`.

### 4.2 Cosa viene salvato nello storico operativo

SQLite `our_history.sqlite` (schema in `history.py`): chiavi partita, pick/action/score, `ev_cons`/`ev_sharp`, probability, quota, agreement, data_factors JSON, hit/result/goals, campi CLV, flag `synthetic_backfill`, freeze Telegram (`score_locked`), id/stats Sofascore.

Altri journal: `telegram_score_freeze.json`, `local_settles.json`, export CSV giocate/successo mensile.

**Paper ROI** (`paper_stats.paper_trading_report`): flat ROI, ROI @ quote, equity Kelly, CLV, beat-close — **solo vote ≥8**; se freeze Telegram, usa lo score/quota **notificati**, non l’ultimo ricalcolo pre-KO.

### 4.3 Metriche di errore e monitoraggio

| Layer | Metriche |
|-------|----------|
| Train OOF / holdout | log-loss, Brier, ECE, accuracy, AUC-OVR → `metrics.json`, `market_metrics.json` |
| Conformal | ampiezza/coverage prediction set |
| Residual EV | RMSE + walk-forward RMSE |
| Online | ROI recente, CLV medio, beat close %, `min_ev` drift |
| Digest | `learn_digest.md` / `online_learn_report.json` |

Non c’è MLflow/W&B: il monitoraggio è **file-based** e report Telegram settimanale.

### 4.4 Limiti strutturali (overfitting, drift, finestre)

**Mitigazioni già presenti**

- Split temporale e walk-forward OOF (default 5 fold).
- Feature solo pre-match; mercato closing come feature (potere predittivo alto ma rischio di “seguire il book”).
- Blend ML–DC–MC e probabilità conservative (`p_cons`) contro overconfidence.
- Residual e pesi con validazione walk-forward.
- Recency e phase-out backfill.
- Separation of concerns: scrapers tattici fuori dall’EV grezzo.

**Rischi aperti**

1. **Concept drift non rilevato esplicitamente** — nessun PSI/ADWIN/monitoring di shift distribuzione feature o odds; si dipende dal retrain calendariale e dai filtri online.
2. **Online learn ≠ aggiornamento del classificatore** — uno shift strutturale nei gol/mercati richiede il job XGB; i filtri possono solo stringere/allentare.
3. **Campione paper ROI ancora sottile** — `min_ev_play` adattivo e claim economici sono ad alta varianza finché n(voto≥8) non raggiunge 30–40.
4. **Backfill synthetic** — utile al bootstrap ma è un lieve leakage (modello su passato); va a phase-out, ma finché presente distorce residual/pesi.
5. **Dual state local vs GHA** — cache Actions + artefatti vs SQLite locale: rischio desync senza `--pull-model` / merge settle journal.
6. **Scrapers soft-fail** — quote/contesto stale senza hard fail → predizioni su feature incomplete (`context_partial`).
7. **Feature window fissa (5)** e cluster latam deboli — underfitting/overfitting localizzati per lega.
8. **Seed MC fisso** — riproducibilità buona, ma stessa sequenza RNG per tutte le partite a pari seed/n; per ranking relativo è accettabile, per stime coda rare con n=400 (cloud) è rumoroso.

---

## 5. Mappa dei Moduli e delle Dipendenze

### 5.1 Albero logico

```text
football-predictor/
├── app.py, main.py                 # UI + orchestrazione CLI
├── requirements.txt / requirements-cloud.txt
├── docs/APPRENDIMENTO.md           # spiegazione learn loop
├── PROJECT_BRIEF_football.md, TECH_ROADMAP.md
├── soccerdata-master/              # libreria scrapers patchabile
├── data/{raw,processed,models}/
├── scripts/                        # cloud_bootstrap, cloud_learn, notify, probes
├── .github/workflows/              # train, odds, telegram, keep-alive
└── modules/
    ├── data_update/                # download, odds, context, history, settle, upcoming
    ├── dataset_loader/             # unificazione CSV → matches
    ├── feature_engineering/        # FEATURE_COLS + stato incrementale
    ├── model_training/             # train, market_models, league_clusters
    ├── predictor/                  # predict, poisson, lambda_utils
    ├── montecarlo/                 # simulate, extras
    ├── calibration/                # metrics, calibrate, conformal, backtest, config
    ├── advisor/                    # advise, value, staking, online_learn, tactics, …
    ├── notify/                     # telegram, alerts, stale_settle
    ├── tipsters/                   # Forebet/PredictZ/Vitibet (quadro)
    ├── sportly_sim/                # viz simulazione
    └── visualization/              # mplsoccer profiles
```

### 5.2 File chiave e responsabilità

| File | Responsabilità |
|------|----------------|
| `main.py` | Pipeline update/train/predict/odds/pull |
| `app.py` | Streamlit: calendario, consigli, valutazione, paper ROI |
| `modules/data_update/download.py` | football-data.co.uk download |
| `modules/data_update/upcoming.py` | Costruzione calendario predizioni |
| `modules/data_update/history.py` | SQLite archive, settle, CLV, export |
| `modules/data_update/history_backfill.py` | Backfill synthetic ricco |
| `modules/data_update/asian_odds.py` / `odds_api.py` / `betfair.py` / `kambi_football.py` | Quote multi-book |
| `modules/data_update/*_context.py` | Contesto tattico/xG esterno |
| `modules/feature_engineering/features.py` | Matrice feature training/inferenza |
| `modules/model_training/train.py` | Train 1X2 + OOF + cluster |
| `modules/model_training/market_models.py` | O/U 2.5 + AH0 |
| `modules/model_training/league_clusters.py` | Routing cluster |
| `modules/predictor/predict.py` | Inferenza ensemble |
| `modules/predictor/poisson.py` | Dixon–Coles |
| `modules/predictor/lambda_utils.py` | λ attesi |
| `modules/montecarlo/simulate.py` | Motore MC |
| `modules/montecarlo/extras.py` | Cards/corners/shots |
| `modules/advisor/advise.py` | Mercati → pick/voto |
| `modules/advisor/value.py` | Edge/EV conservativo |
| `modules/advisor/staking.py` | Kelly / no-bet |
| `modules/advisor/online_learn.py` | Loop post-settle |
| `modules/advisor/learn_policy.py` | Policy trainable/ROI |
| `modules/advisor/residual_ev.py` | Correzione EV |
| `modules/advisor/paper_stats.py` | Paper trading report |
| `modules/advisor/tactics.py` | Analisi tattica (display) |
| `modules/calibration/*` | Calibrazione, conformal, metriche |
| `modules/notify/telegram.py` | Alert + freeze |

### 5.3 Grafo dipendenze (inferenza consiglio)

```text
DatasetLoader / FeatureEngineer
        ↓
ModelTrainer ──► best_model / market_models / calibration
        ↓
MatchPredictor ──► λ + proba ML
        ↓
MonteCarloSimulator ──► mercati
        ↓
advise ← odds caches ← asian/odds_api/betfair/kambi
   ├── value.enrich_value ← calibration / residual
   ├── staking.quarter_kelly / no_bet
   ├── tactics / data_signal / quadro / tipsters  (voto, non EV core)
   └── output → upcoming.json / Telegram / Streamlit
```

---

## 6. Consigli di Ottimizzazione

Quattro interventi concreti, ordinati per impatto su robustezza dell’auto-apprendimento e delle probabilità.

### 6.1 Chiudere il gap “online learn ↔ modello base” con un retrain **condizionato al drift**

Oggi il retrain XGB è calendariale; l’online learn muove solo filtri.  
**Proposta:** dopo ogni `learn_from_settled`, calcolare indicatori di drift (ECE rolling 1X2 su settle live, Brier O/U, CLV medio, PSI sulle feature `mkt_*` / `elo_diff`) e, se soglie superate per N giorni consecutivi, triggerare un **retrain incrementale** (o almeno un job GHA `workflow_dispatch`) invece di attendere solo il cron.  
Così il sistema reagisce al concept drift senza retrain inutili quando i filtri bastano.

### 6.2 Calibrazione **per mercato e per cluster**, non solo 1X2 globale

I cluster latam e alcuni Big5 mostrano ECE alto; O/U/AH hanno temperature separate ma i reliability bin online sono centrati sul 1X2.  
**Proposta:** estendere `reliability_*` e `online_p_factor` a **O/U 2.5 e AH0 per cluster** (con shrinkage verso globale quando n&lt; soglia), e usare quei fattori dentro `enrich_value` prima di `p_cons`.  
Riduce miscalibrazione sistematica che oggi viene “nascosta” alzando `min_ev_play`.

### 6.3 Monte Carlo adattivo + seed per-match

Con `n_sims=400` (cloud) le code combo sono rumorose; con 10k e seed globale fisso si spreca calcolo su partite ovvie.  
**Proposta:** (a) seed = hash(`match_key`) per indipendenza tra fixture; (b) **stopping adattivo** (es. continua finché SE di `p_home` e `p_over_25` &lt; ε, con min/max bound); (c) allineare cloud calendar a almeno 2–4k sims sulle partite candidate al voto ≥7.  
Migliora stabilità delle probabilità usate da EV senza costi lineari fissi.

### 6.4 Curriculum del residual EV e del paper ROI su **solo live rich + freeze Telegram**

Finché il residual Ridge vede migliaia di sample gonfiati da replicate/backfill, può ottimizzare un mondo diverso dal paper ROI (voto≥8, quote reali).  
**Proposta:** dual-fit esplicito — (1) residual “ampio” per stabilità, (2) residual “production” fit-only su `is_roi_eligible` / freeze, usato in prod quando n≥30; loggare sempre entrambi RMSE WF in `online_learn_report.json`. Parallelamente, gate più duro: non allentare `min_ev_play` finché non si raggiunge il livello “Validazione economica” (n≥30).  
Allinea auto-apprendimento economico al campione che conta davvero.

---

## Appendice — Confini di design (cosa entra nell’EV)

| Componente | Entra in EV / Kelly? |
|------------|----------------------|
| XGB/RF + Dixon–Coles + MC + market XGB | **Sì** |
| Temperature, reliability bins, online_p_factor, p_cons | **Sì** (via probabilità conservative) |
| Residual EV | Soft filter / aggiustamento |
| Tactics, ClubElo display, data_signal, Sportly-sim, tipster | **No** — quadro / voto / UI |

---

*Fine dell’analisi. Per il dettaglio operativo dell’apprendimento online si veda anche `docs/APPRENDIMENTO.md`; per la roadmap prodotto `TECH_ROADMAP.md` e `PROJECT_BRIEF_football.md`.*
