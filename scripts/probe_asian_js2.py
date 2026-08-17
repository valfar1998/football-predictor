import re
import urllib.request

js = urllib.request.urlopen(
    urllib.request.Request(
        "https://www.asianbetsoccer.com/settings/onloaded_V4.js",
        headers={"User-Agent": "Mozilla/5.0"},
    ),
    timeout=30,
).read().decode("utf-8", "replace")

for kw in ["tablenext", "tablematch", "ajax", "post(", "get(", "xmlhttp", "book_filter", "value_next"]:
    idx = js.lower().find(kw.lower())
    print(kw, idx)
    if idx >= 0:
        print(js[max(0, idx - 80) : idx + 200])
        print("---")

# strings that look like endpoints
for m in re.finditer(r"[a-zA-Z0-9_./-]{4,80}", js):
    s = m.group(0)
    if any(x in s.lower() for x in ("table", "match", "next", "data", "load", "result", "odds")):
        if "/" in s or "tab" in s.lower():
            pass

hits = set()
for pat in [
    r"url\s*:\s*['\"]([^'\"]+)['\"]",
    r"\.post\(['\"]([^'\"]+)['\"]",
    r"\.get\(['\"]([^'\"]+)['\"]",
    r"gtc\s*\+\s*['\"]([^'\"]+)['\"]",
    r"['\"](/[a-zA-Z0-9_./-]+)['\"]",
]:
    for m in re.finditer(pat, js):
        hits.add(m.group(1))

print("paths", sorted(hits)[:50])
