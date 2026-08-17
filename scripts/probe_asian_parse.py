import re
import urllib.request
import time

url = f"https://botbot3.space/tables/v4/Q/tablenext/day0/12fa2eba2655cdc08a0d92fd601c498da2f49b54.js?date={int(time.time()*1000)}"
data = urllib.request.urlopen(
    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.asianbetsoccer.com/it/nextgame.html"}),
    timeout=30,
).read().decode("utf-8", "replace")
open("data/raw/asian_day0_sample.js", "w", encoding="utf-8").write(data)
print("len", len(data))
# find function calls
for fn in ["getDatanext", "getData", "Datalast", "Datanext"]:
    print(fn, data.count(fn))
calls = re.findall(r"getDatanext\d?\([^;]{0,400}\)", data)
print("calls", len(calls))
if calls:
    print(calls[0][:500])
    print("---")
    print(calls[1][:500] if len(calls)>1 else "")
