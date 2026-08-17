import urllib.request
import time

base = "https://botbot3.space/tables/v4"
sdm = int(time.time() * 1000)
books = {
    "bet365": "12fa2eba2655cdc08a0d92fd601c498da2f49b54",
    "avg": "f742dd97165c680c4b28d22dd56d0567189f8e3d",
    "188": "6161483bb3095c88f7e154d712b5decd13c888f4",
    "default": "4b73a4a8479c5190f625fbb91ff8bb9b917c3ae2",
}
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.asianbetsoccer.com/it/nextgame.html",
}
for gs in ["Q", "A", "L", "it", "en", "1", "0"]:
    for day in ["tablenext/day0", "tablenext/day1", "tablenext/day2"]:
        for bname, bhash in books.items():
            url = f"{base}/{gs}/{day}/{bhash}.js?date={sdm}"
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=20
                ).read().decode("utf-8", "replace")
                if len(data) > 300 and "nomatch" not in data[:200].lower():
                    print("OK", gs, day, bname, "len", len(data))
                    print(data[:1200])
                    print("...")
                    raise SystemExit(0)
            except Exception as e:
                if "HTTP Error" in str(e):
                    pass
print("none")
