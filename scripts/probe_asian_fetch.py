import re
import urllib.request
import time

js = urllib.request.urlopen(
    urllib.request.Request(
        "https://www.asianbetsoccer.com/settings/tablefunc.v5.book.min.js",
        headers={"User-Agent": "Mozilla/5.0"},
    ),
    timeout=30,
).read().decode("utf-8", "replace")

for name in ["function CookieStats", "function CookieBook", "function bb"]:
    idx = 0
    while True:
        idx = js.find(name, idx)
        if idx < 0:
            break
        print(js[idx : idx + 400])
        print("---")
        idx += 1

# try fetch sample data js
base = "https://www.asianbetsoccer.com/tables/v4"
sdm = int(time.time() * 1000)
books = {
    "bet365": "12fa2eba2655cdc08a0d92fd601c498da2f49b54",
    "avg": "f742dd97165c680c4b28d22dd56d0567189f8e3d",
    "188": "6161483bb3095c88f7e154d712b5decd13c888f4",
}
for gs in ["Q", "A", "L", "it", "en"]:
    for day in ["tablenext/day0", "tablenext/day1"]:
        for bname, bhash in books.items():
            url = f"{base}/{gs}/{day}/{bhash}.js?date={sdm}"
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.asianbetsoccer.com/it/nextgame.html"}),
                    timeout=20,
                ).read().decode("utf-8", "replace")
                if len(data) > 200 and "404" not in data[:100]:
                    print("OK", url, "len", len(data))
                    print(data[:800])
                    raise SystemExit
            except Exception as e:
                pass
print("no url worked")
