import re
import urllib.request

for url in [
    "https://www.asianbetsoccer.com/it/nextgame.html",
    "https://www.asianbetsoccer.com/it/",
]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    print("URL", url, "len", len(html))
    for pat in ["api/", ".json", "table", "C1", "login", "Accedi", "nextGame", "match", "1X2"]:
        print(" ", pat, html.lower().count(pat.lower()))
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', html)[:10]
    print(" scripts", scripts)
    apis = [a for a in re.findall(r"https?://[^\s\"']+", html) if "api" in a.lower() or "service" in a.lower()]
    print(" api urls", apis[:20])
