# football-predictor

Previsione risultati calcistici (1X2 + over/under) con feature di forma/xG, Random Forest / XGBoost e simulazione Monte Carlo.

```powershell
cd C:\Users\valba\Desktop\corsi\football-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --train
python main.py --predict "Inter" "Milan"
```

I prompt usati per i moduli sono in `docs/prompts.md`.
