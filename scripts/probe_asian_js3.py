import re
import urllib.request

js = urllib.request.urlopen(
    urllib.request.Request(
        "https://www.asianbetsoccer.com/settings/onloaded_V4.js",
        headers={"User-Agent": "Mozilla/5.0"},
    ),
    timeout=30,
).read().decode("utf-8", "replace")

for m in re.finditer(r"tables/v4[^'\"]{0,120}", js):
    print(m.group(0))

# find function building url
idx = js.find("tables/v4")
while idx != -1:
    print("CTX:", js[max(0, idx - 150) : idx + 250])
    print("====")
    idx = js.find("tables/v4", idx + 1)
