import requests, urllib3
urllib3.disable_warnings()
from bs4 import BeautifulSoup

url = "https://rbi.org.in/Scripts/BS_ViewBulletin.aspx"
res = requests.get(url, verify=False, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(res.text, "html.parser")

found = False
for a in soup.find_all("a", href=True):
    text = a.text.lower()
    if "finance" in text or "public limited" in text or "companies" in text:
        print(f"{a.text.strip()} -> {a['href']}")
        found = True

if not found:
    print("No matching articles found on the main bulletin page.")
