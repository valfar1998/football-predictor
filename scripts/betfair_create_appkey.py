"""Crea la Betfair Delayed App Key via API diretta (senza portal web).

Uso:
    python scripts/betfair_create_appkey.py
"""
import json
import urllib.request
import urllib.parse
import getpass

LOGIN_URL = "https://identitysso.betfair.it/api/login"
ACCOUNT_URL = "https://api.betfair.com/exchange/account/json-rpc/v1"

def login(username: str, password: str) -> str:
    data = urllib.parse.urlencode({"username": username, "password": password}).encode()
    req = urllib.request.Request(
        LOGIN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "X-Application": "1",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read().decode())
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Login fallito: {result.get('error') or result}")
    token = result.get("token")
    print(f"Login ok. Session token: {token[:20]}...")
    return token

def get_app_keys(token: str) -> list:
    payload = [{"jsonrpc": "2.0", "method": "AccountAPING/v1.0/getDeveloperAppKeys", "params": {}, "id": 1}]
    req = urllib.request.Request(
        ACCOUNT_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Authentication": token,
            "X-Application": "1",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read().decode())
    return result[0].get("result") or []

def create_app_key(token: str, name: str) -> dict:
    payload = [{"jsonrpc": "2.0", "method": "AccountAPING/v1.0/createDeveloperAppKeys", "params": {"appName": name}, "id": 1}]
    req = urllib.request.Request(
        ACCOUNT_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Authentication": token,
            "X-Application": "1",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read().decode())
    r = result[0]
    if "error" in r:
        raise RuntimeError(f"Errore creazione: {r['error']}")
    return r.get("result") or {}

if __name__ == "__main__":
    print("=== Betfair App Key Creator ===")
    username = input("Username Betfair: ").strip()
    password = getpass.getpass("Password Betfair: ")

    token = login(username, password)

    # Prima controlla se esiste già
    keys = get_app_keys(token)
    if keys:
        print("\nApp Key già esistenti:")
        for app in keys:
            print(f"  Nome: {app.get('appName')}")
            for k in app.get("appVersions") or []:
                label = "Delayed" if k.get("delayData") else "Live"
                key = k.get("applicationKey") or k.get("delayedAppKey") or k.get("ownerAppKey")
                print(f"  {label} Key: {key}  active={k.get('active')}")
    else:
        print("\nNessuna chiave trovata. Creo 'FootballPredictor'...")
        result = create_app_key(token, "FootballPredictor")
        print(f"  Nome: {result.get('appName')}")
        for k in result.get("appVersions") or []:
            label = "Delayed" if k.get("delayData") else "Live"
            key = k.get("applicationKey") or k.get("delayedAppKey") or k.get("ownerAppKey")
            print(f"  {label} Key: {key}  active={k.get('active')}")

    input("\nPremi Invio per chiudere...")
