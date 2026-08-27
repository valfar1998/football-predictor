# football-predictor

App di **analisi e value betting** pre-match sul calcio: probabilità da ML + Monte Carlo, confronto con quote sharp/Asian, voto unificato e apprendimento dalle partite chiuse.

**Documentazione completa:** [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) (architettura, fonti dati, apprendimento) · [`TECH_ROADMAP.md`](TECH_ROADMAP.md) (cosa è fatto / cosa resta).

---

## Avvio rapido

```powershell
cd C:\Users\valba\Desktop\corsi\football-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --train          # prima volta o dopo settimane offline
streamlit run app.py            # oppure: python main.py --ui
```

### Uso quotidiano (UI)

| Bottone | Quando usarlo | Tempo tipico |
|---------|---------------|--------------|
| **Scarica modello da GitHub** | Preferito: ultimo train da Actions (no train locale) | ~1–2 min |
| **Solo quote e calendario** | Quote mosse / nuove partite fd, **senza** riallenare | ~1–3 min |
| **Aggiorna dati + modello** | Train completo in locale (se non usi GitHub) | ~1h+ |
| **Apprendi da partite chiuse** (tab Valutazione) | Dopo settle, ogni 1–2 giorni (spesso già auto) | pochi secondi |

```powershell
# modello cloud → PC (serve: gh auth login)
python main.py --pull-model
# opzionale: subito dopo ricostruisci il calendario (lento, Monte Carlo pieno)
python main.py --pull-model --rebuild-calendar
```

---

## In sintesi: da dove arrivano i dati

```text
STORICO (train)                    LIVE / CALENDARIO (pronostico)
─────────────────                  ───────────────────────────────
football-data.co.uk    ──┐         fixtures + quote 1X2 (fd, org, mondo)
  risultati + HY/HC      │         AsianBetSoccer, Pinnacle, Betfair
FBref / Understat        ├──►      ClubElo, meteo, tipster
  xG, stile, player      │         FBref, FotMob, StatsBomb, Sofascore
ClubElo, side_rates      │              │
                         ▼              ▼
              features.csv + best_model.joblib
                         │
                         ▼
              MatchPredictor + Monte Carlo + advise (EV/Kelly/voto)
                         │
                         ▼
              upcoming_predictions.json  ──►  archive SQLite (pre-match)
                         │                         │
                    partita giocata                 ▼
                         └──────────────────►  settle (hit/miss)
                                                   │
                                                   ▼
                                    online_learn (calibrazione, residual, pesi)
```

**Regola d’oro:** le fonti “fragili” (live, scraping) alimentano **quadro e voto**; **EV e Kelly** usano solo probabilità calibrate + quote verificabili.

---

## Stato attuale (2026-08-22)

Snapshot dopo aggiornamento con partite chiuse:

| Metrica | Valore | Gate roadmap |
|---------|--------|--------------|
| Partite in storico SQLite | 3 892 | — |
| **Settled** (esito noto) | **770** | ≥ 30 per peso storico nel voto ✅ |
| Settled con EV (`ev_cons` / `ev_sharp`) | 114 | ≥ 80 per **residual EV** ✅ |
| **Storico ricco** (quota + EV + `data_factors` + accordo) | **51** | target **80** (~64%) |
| Residual EV in produzione | sì (WF-RMSE ≈ 0,51) | RMSE ≤ 0,55 ✅ |
| `online_p_factor` 1X2 | 74 / 80 settled | quasi attivo |

**Conclusione:** il **codice e il nucleo ML** sono pronti; il **residual EV è già in produzione**. Per sbloccare al 100% l’apprendimento “ricco” (paper Kelly @ quote reali, bin online maturi, pesi ottimali) servono **~29 pick pre-match in più** archiviati **prima** del calcio d’inizio, con quote reali e pick coperto dal modello — si accumulano con refresh quotidiani.

Dettaglio gate e prossimi passi: [`TECH_ROADMAP.md`](TECH_ROADMAP.md).

---

## CLI utili

```powershell
python main.py --pull-model          # modello da GitHub Actions (no train locale)
python main.py --update              # download + train + calendario (come bottone UI)
python main.py --odds-update         # quote + calendario leggero (no mondiale/coppe/tipster)
python main.py --train-markets       # solo modelli O/U 2.5 e AH 0
python main.py --predict "Inter" "Milan"
```

Prompt modulari storici: `docs/prompts.md`.
