import re
import urllib.request

for u in [
    "https://www.asianbetsoccer.com/settings/tablefunc.v5.book.min.js",
    "https://www.asianbetsoccer.com/settings/onloaded_V4.js",
]:
    js = urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30
    ).read().decode("utf-8", "replace")
    print("---", u, "len", len(js))
    for m in re.finditer(r"/[a-zA-Z0-9_./-]+\.(php|ashx|json|html)", js):
        print(" path", m.group(0))
