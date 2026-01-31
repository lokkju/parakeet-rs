# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "parfive",
# ]
# ///

print("""
Fetching examples of Harvard Sentances from the Open Speech Repository
""")
BASE_URL = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_00{}_8k.wav"

from parfive import Downloader

urls = [BASE_URL.format(x) for x in list(range(10,19)) + list(range(30,32)) + list(range(34,40)) + list(range(57,61))]

dl = Downloader()
for url in urls:
    dl.enqueue_file(url, path="./")
files = dl.download()
