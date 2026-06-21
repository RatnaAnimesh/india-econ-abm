import requests
import json
import os

def download_mca_maharashtra():
    url = "https://api.data.gov.in/resource/4dbe5667-7b6b-41d7-82af-211562424d9a"
    params = {
        "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
        "format": "json",
        "filters[CompanyStateCode]": "maharashtra",
        "limit": "1000"
    }
    print("Downloading MCA Maharashtra Sample...")
    res = requests.get(url, params=params)
    if res.status_code == 200:
        data = res.json()
        os.makedirs("data/raw", exist_ok=True)
        with open("data/raw/mca_maharashtra_sample.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Downloaded sample of 1000 rows to data/raw/mca_maharashtra_sample.json")
        print(f"Total companies in Maharashtra available: {data.get('total', 'Unknown')}")
    else:
        print("Failed:", res.status_code, res.text)

if __name__ == "__main__":
    download_mca_maharashtra()
