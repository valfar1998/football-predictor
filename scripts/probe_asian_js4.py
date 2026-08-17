import re
import urllib.request

js = urllib.request.urlopen(
    urllib.request.Request(
        "https://www.asianbetsoccer.com/settings/onloaded_V4.js",
        headers={"User-Agent": "Mozilla/5.0"},
    ),
    timeout=30,
).read().decode("utf-8", "replace")

for name in ["function bb", "var gS", "gS =", "sdm", "function get_date_offset", "tablenext/day", "bookvalue", "4b73a4a8479c5190f625fbb91ff8bb9b917c3ae2"]:
    idx = js.find(name)
    print(name, idx)
    if idx >= 0:
        print(js[idx : idx + 500])
        print("---")
