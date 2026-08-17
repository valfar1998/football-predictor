# Prompt per generare il sistema Football Predictor

Usa questi prompt in Cursor (Agent mode), un modulo alla volta. Il progetto atteso è:

```
/football-predictor
├── /data/raw
├── /data/processed
├── /data/models
├── /modules/dataset_loader
├── /modules/feature_engineering
├── /modules/model_training
├── /modules/predictor
├── /modules/montecarlo
├── /docs/prompts.md
└── main.py
```

---

## 1. Dataset Loader

Genera un modulo Python che:

- carica dataset calcistici da file CSV
- unisce dataset diversi (Understat, FBref, ecc.)
- normalizza nomi squadre
- gestisce valori mancanti
- salva il dataset pulito in `/data/processed`

Il codice deve essere modulare e riutilizzabile.

---

## 2. Feature Engineering

Genera un modulo Python che esegue feature engineering per partite di calcio, includendo:

- forma recente (ultime 5 partite)
- xG e xGA
- gol fatti/subiti
- fattore casa
- differenza qualità rosa
- variabili temporali

Il modulo deve restituire un DataFrame pronto per il training. Le feature vanno calcolate solo su partite **precedenti** (niente data leakage).

---

## 3. Model Training

Genera un modulo Python che:

- divide il dataset in train/test
- addestra modelli ML (Random Forest, XGBoost)
- valuta accuracy, log-loss e AUC
- salva il modello migliore in `/data/models`

Il codice deve essere pulito e commentato. Preferire lo split temporale all’ultimo 20% delle date.

---

## 4. Predictor

Genera un modulo Python che:

- carica il modello salvato
- riceve come input due squadre
- costruisce le feature necessarie
- restituisce probabilità di:
  - vittoria casa
  - pareggio
  - vittoria ospite

Il modulo deve essere semplice da integrare.

---

## 5. Simulazione Monte Carlo

Genera un modulo Python che:

- riceve le probabilità del predictor
- simula la partita 10.000 volte
- usa distribuzione Poisson per gol
- calcola probabilità finali di ogni risultato
- genera anche probabilità over/under

Il modulo deve essere efficiente e parametrico (NumPy vettorizzato).

---

## 6. Orchestratore (`main.py`)

Genera un file `main.py` che:

- carica dataset
- esegue feature engineering
- addestra modelli
- genera prediction
- esegue simulazione Monte Carlo
- salva output finale in JSON

Il file deve coordinare tutti i moduli del progetto.

---

## 7. Documentazione

Genera un file `prompts.md` che contiene tutti i prompt necessari per creare ogni modulo del sistema di prediction calcio. Formattalo in modo ordinato e leggibile.

---

## Vincoli

- Python 3.10+, pandas, scikit-learn, XGBoost, NumPy, joblib
- Nessuno scraping anti-bot; i CSV si caricano da `data/raw`
- Se `data/raw` è vuoto, generare un dataset sintetico di demo
