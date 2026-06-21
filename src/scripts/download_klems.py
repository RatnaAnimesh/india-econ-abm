import requests
import os

url = "https://rbidocs.rbi.org.in/rdocs/content/docs/INDIAKLEMS08072024.xlsx"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print(f"Downloading {url}...")
res = requests.get(url, headers=headers)

if res.status_code == 200:
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/INDIAKLEMS08072024.xlsx", "wb") as f:
        f.write(res.content)
    print("KLEMS downloaded successfully.")
else:
    print("Failed to download KLEMS. Status code:", res.status_code)
