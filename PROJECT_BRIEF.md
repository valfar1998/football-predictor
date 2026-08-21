# Football Predictor — Project Brief (per analisi esterna)

Documento di sintesi del progetto **football-predictor** (repo locale / GitHub `valfar1998/football-predictor`).  
Scopo: dare a un altro modello / analista contesto sufficiente per suggerire miglioramenti, senza dover leggere tutto il codice.

**Lingua UI/note:** italiano. **Stack:** Python 3.10+, Streamlit, pandas, scikit-learn, XGBoost, soccerdata, statsbombpy, mplsoccer.

---

## 1. Cos’è

App di **analisi e value betting** sul calcio (pre-match), non un bookmaker né un tipster automatico.

Pipeline tipica:

1. Scarica risultati storici + calendari + quote  
2. Costruisce feature (forma, xG rolling, Elo, casa/trasferta, quote implicite, …)  
3. Allena modello ML + stima λ Poisson / Dixon–Coles  
4. Simula Monte Carlo → probabilità 1X2 / O-U / mercati derivati  
5. Confronta con quote (Asian, Pinnacle/Odds API, Betfair, football-data) → EV / Kelly / voto  
6. Mostra **quadro** multi-fonte + **validazione** (non EV) + UI Streamlit  

**Principio di design importante:** molte fonti arricchiscono il *quadro* e il *voto unificato*, ma **non devono contaminare EV / Kelly / p_cons** (probabilità “conservative” usate per value).

---

## 2. Struttura cartelle

```
football-predictor/
├── app.py                 # UI Streamlit (calendario, mercati, singola, eval)
├── main.py                # CLI: --train, --predict, refresh quote, notify
├── requirements.txt
├── data/
│   ├── raw/               # token/key, asian_odds, cache quote (gitignored secrets)
│   ├── processed/         # features, upcoming_predictions, context CSV, SQLite history
│   └── models/            # joblib + calibrazione JSON
├── modules/
│   ├── data_update/       # download, odds, context (FBref, Understat, FotMob, …)
│   ├── feature_engineering/
│   ├── model_training/
│   ├── predictor/         # predict.py, poisson.py
│   ├── montecarlo/
│   ├── advisor/           # advise, value, quadro, validation, data_signal, tactics
│   ├── sportly_sim/       # simulazione sintetica stile Sportly (non live)
│   ├── calibration/
│   ├── tipsters/
│   ├── notify/            # Telegram alerts
│   └── visualization/
├── scripts/               # probe, bootstrap, VBS refresh nascosto
└── .github/workflows/     # refresh / telegram (Betfair spesso 403 da datacenter)
```

---

## 3. Modello predittivo (nucleo)

| Pezzo | Ruolo |
|--------|--------|
| `FeatureEngineer` | Forma punti/GD, xG/xGA rolling, WR casa/trasferta, riposo, Elo, features di mercato `mkt_p_*` |
| `ModelTrainer` | XGBoost (+ ensemble) su 1X2 |
| `poisson` / Dixon–Coles | λ casa/trasferta; blend Understat xG quando c’è; aggiustamento meteo Open-Meteo |
| `MonteCarloSimulator` | Distribuzione scoreline → mercati |
| Calibrazione | Temperature scaling (anche per-lega), soglie EV |

**Input tipici predizione:** home, away, kickoff, league, odds, weather, context xG esterni.

**Output tipici:** `model_probabilities`, `expected_goals` (λ), `features`, `montecarlo`, ensemble meta.

---

## 4. Advisor (value + voto)

File chiave: `modules/advisor/{advise,value,validation,quadro,data_signal,tactics}.py`.

### 4.1 Value / EV
- Confronta probabilità modello vs quote reali  
- EV “cons” (conservativo) vs EV “sharp” (Pinnacle/Betfair se presenti)  
- Kelly frazionario (¼ tipico)  
- Cap voto probabilità dinamici per volatilità di lega  
- Allineamento movimento Asian (apertura → attuale) scala il voto mercato  

### 4.2 Voto unificato (`meta_analysis` / `score_unified`)
Pesi su gambe: **value, kelly, asian, workflow, history, combos**.  
Poi Δ da validazione (stadio, tattica, ML↔MC, forma, Sportly-sim, **analisi dati**).  
`no_bet` / senza modello → voto limitato.

### 4.3 Quadro (fonti ortogonali)
Ogni fonte dà un lean 1X2 + nota, **senza mescolarle nell’EV**:

- Monte Carlo, Modello ML  
- Quote / mercato implicito  
- ClubElo  
- λ Poisson (forma)  
- FBref, Understat, StatsBomb, Sofascore, **FotMob**  
- Sportly-sim (sintetico)  
- **Analisi dati** (fusione fattori)  
- Storico locale (SQLite esiti nostri)  
- Tipster pubblici, Steam Asian, Meteo, Validazione  

### 4.4 Analisi dati (`data_signal.py`) — recente
Algoritmo esplicito che pesa:

1. Forma rolling  
2. Casa/trasferta WR  
3. xG rolling feature  
4. Understat  
5. Classifica FotMob (se `played≥1`) altrimenti Sofascore  
6. FBref GD/pg  
7. StatsBomb (peso basso)  
8. Riposo  
9. Elo  

→ edge, lean, confidenza, breakdown fattori.  
Micro Δ voto (±0.25 / ±0.5) se confidenza alta e allinea/contraddice il pick. **Non tocca EV.**

---

## 5. Fonti dati (stato integrazione)

### Già integrate (vere API o dataset)
| Fonte | Uso |
|--------|-----|
| football-data.co.uk | Risultati storici, Avg odds |
| football-data.org | Coppe / fixtures ufficiali |
| TheSportsDB | Coppe, metadata, venues |
| API-Football | Coppe / world fixtures (free tier limitato) |
| OpenLigaDB + OpenFootball | Calendario mondiale |
| The Odds API | Pinnacle (free tier) |
| Betfair Exchange | Quote (GHA spesso 403 IT/datacenter; locale OK con app key) |
| AsianBetSoccer | Quote Asian + movimento (scraping sito) |
| ClubElo | Elo squadre |
| FBref via soccerdata | Contesto team / contrib giocatori |
| Understat via soccerdata | xG Big 5 (+ RPL dove coperto) |
| StatsBomb open data | Event-level aggregato team |
| Sofascore via soccerdata | Classifica attuale Big 5 |
| WhoScored | Assenze (lento, preview limitata) |
| Open-Meteo | Meteo città stadio → λ adj |
| **FotMob `/api/data`** | Classifiche + calendario 7g (non ufficiale, no key) |

### Simulato / interno
| Fonte | Note |
|--------|------|
| `sportly_sim` | xG/momentum/pressione/shotmap **sintetici** da λ + FBref; non è live FotMob |

### Esplicitamente evitati / non vendorizzati
- SDK `sportly-master` (eliminato)  
- Django `Public-FotMob-API` (eliminato)  
- Softascore curl_cffi WAF bypass aggressivo  
- Transfermarkt / XI live ufficiali gratis  
- soccerapi scraper multi-book come dipendenza stabile  
- worldfootballR (R)  

### Wishlist nota (non fare duplicati)
StatsBomb Impect packing, Kaggle bulk, ProphitBet come benchmark modelli, Odds API più book, live FotMob matchDetails in bulk (rate-limit).

---

## 6. UI Streamlit (`app.py`)

Tab principali:

1. **Calendario** — filtri, voto, consigli, dettagli + / quadro / validazione / analisi dati  
2. **Tutti i mercati** — flat markets con filtri EV/voto/quote  
3. **Singola partita** — predict on demand  
4. **Valutazione** — calibrazione / bankroll path  

**Filtro date:** default da **oggi** → ultima data disponibile (chiave widget con `YYYY-MM-DD` così si aggiorna ogni giorno).  
Sidebar: refresh dati, token, contesto extra (FBref, Understat, StatsBomb, Sofascore, FotMob, WhoScored).

---

## 7. CLI / automazione

```text
python main.py --train
python main.py --predict "Home" "Away"
# refresh quote + calendari + context (vari pipeline in main.py)
```

- `scripts/notify_refresh_hidden.vbs` — refresh nascosto Windows  
- GitHub Actions — alert Telegram / refresh (attenzione Betfair 403)  
- Secrets: `.env` + `data/raw/*.key|*.token` (gitignored)  

---

## 8. Cosa entra nel modello vs solo analisi

| Segnale | Feature ML / λ | EV/Kelly | Quadro / voto |
|---------|----------------|----------|----------------|
| Forma, xG rolling, Elo, rest | sì | indiretto via p | sì |
| Understat xG | blend λ | indiretto | sì |
| Quote implicite `mkt_p_*` | sì (dopo retrain) | confronto | sì |
| Meteo | λ adj | indiretto | sì |
| FBref / Sofascore / FotMob / SB | no (o marginale) | no | sì |
| Sportly-sim / data_signal | no | no | sì + Δ voto |
| Asian steam | no | allineamento voto | sì |
| Tipster | no | no | sì / leg voto |

---

## 9. Limiti noti / rischi di fragilità

1. **FotMob / Sofascore / FBref** non ufficiali o via soccerdata → rotture HTML/API possibili.  
2. Inizio stagione: classifiche con `played=0` → lean classifica assente; restano forma/xG storici.  
3. **Betfair** da CI spesso bloccato.  
4. Free tier **API-Football** (100 req/giorno) e **Odds API** stretti.  
5. WhoScored lento / copertura ridotta.  
6. Storico locale entra nel voto solo dopo abbastanza partite chiuse in SQLite.  
7. XI / infortuni live di qualità non disponibili gratis in modo stabile.  
8. `data_signal` è euristico pesato (non ML second-stage); pesi non calibrati su ROI.  

---

## 10. Idee per chi analizza da fuori (hints)

Suggerimenti utili da valutare / implementare:

### A. Modello
- Second-stage calibrator: `data_signal.edge` + market move → residual EV  
- Separare leghe / coppe (prior draw diversi)  
- Feature “days into season” e “sample size” per smorzare PPG precoci  
- Conformal prediction / intervalli su 1X2  

### B. Dati
- FotMob `matchDetails` on-demand solo per top-N picks (lineup flag, xG H2H recente)  
- Rolling xG da partite finite FotMob (cache, rate limit)  
- StatsBomb open: più stagioni / mapping nomi più aggressivo  
- OpenFootball stadi → geocode meteo più completo  

### C. Advisor
- Pesi `data_signal` ottimizzati su backtest Brier / ROI (non a occhio)  
- Quadro: score di **accordo fonti** già presente → usarlo come filtro no_bet più forte  
- Distinguere mercati 1X2 vs AH/O-U nel data_signal  

### D. Prodotto
- Export CSV/JSON del calendario filtrato per analisi esterna  
- Dashboard “solo disaccordi” (dati vs mercato vs ML)  
- Paper trading log strutturato già parzialmente in history SQLite  

### E. Non fare (di solito)
- Mischiare FotMob live nell’EV senza calibrazione  
- Dipendere da un solo scraper fragile per il core  
- Gonfiare il voto quando mancano quote reali  

---

## 11. File “leggi questi prima”

Ordine consigliato per un altro agente:

1. `PROJECT_BRIEF.md` (questo file)  
2. `modules/advisor/advise.py` + `value.py`  
3. `modules/advisor/quadro.py` + `data_signal.py` + `validation.py`  
4. `modules/predictor/predict.py` + `poisson.py`  
5. `modules/data_update/upcoming.py` + `download.py`  
6. `modules/data_update/fotmob_context.py`  
7. `app.py` (filtri + rendering)  
8. `main.py` (pipeline entrypoints)  

---

## 12. Comandi rapidi

```powershell
cd C:\Users\valba\Desktop\corsi\football-predictor
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --train
python main.py --predict "Inter" "Milan"
streamlit run app.py
# oppure
python main.py --ui
```

Contesto FotMob (solo quadro):

```powershell
python -c "from modules.data_update.fotmob_context import download_fotmob_context; print(download_fotmob_context(days=7))"
```

---

## 13. Snapshot decisionale (una frase)

> Previsione ML+MC calibrata per probabilità; quote sharp/Asian per value; contesto multi-fonte (incluso FotMob e analisi dati) per **quadro e fiducia**, non per riscrivere l’edge monetario senza backtest.

---

## 14. Roadmap strutturale — stato implementazione (ago 2026)

| Item | Stato | Note |
|------|--------|------|
| Accordo fonti → no_bet + Δ voto | **fatto** | `modules/advisor/agreement.py` |
| Intervalli probabilità (proxy conformal MC bootstrap) | **fatto** | `montecarlo.prob_intervals`; filtra se fragile |
| `days_into_season` | **fatto** | feature + FEATURE_COLS (serve `--train` per entrare nel ML) |
| data_signal per mercato 1X2 / O/U / AH | **fatto** | `data_signal.markets` |
| Residual EV second-stage | **scaffold** | `residual_ev.py`; fit da SQLite ≥40 settled; filtra/voto, non riscrive p_cons |
| FotMob matchDetails top-N (voto≥7) | **fatto** | cache 6h in `fotmob_details_cache.json` |
| Dashboard disaccordi | **fatto** | tab Valutazione |
| Paper trading SQLite | **base** | per lega/voto/pick; ROI flat |
| Modelli separati per lega | **todo** | solo T-scaling oggi |
| Pesi data_signal ottimizzati ROI/Brier | **todo** | ancora euristici; residual aiuta |
| StatsBomb più stagioni / OpenFootball geocode | **todo** | |
| Conformal full (calibration set OOF) | **todo** | ora bootstrap MC |

*Aggiornato con il piano strutturale Priorità 1–3.*
