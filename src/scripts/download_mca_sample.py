import os
from datagovindia import DataGovIndia
import pandas as pd

API_KEY = os.environ.get("DATA_GOV_API_KEY", "YOUR_API_KEY_HERE")

def download_sample():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set DATA_GOV_API_KEY environment variable.")
        return

    data_gov = DataGovIndia(API_KEY)
    
    print("Updating metadata...")
    data_gov.update_metadata()
        
    print("Searching for Company Master Data...")
    results = data_gov.search('Company Master Data')
    
    if not results.empty:
        resource_id = results.iloc[0]['resource_id']
        print(f"Found resource: {resource_id} - {results.iloc[0]['title']}")
        print(f"Downloading resource: {resource_id}")
        data = data_gov.get_data(resource_id)
        
        os.makedirs('data/raw', exist_ok=True)
        data.to_csv(f"data/raw/mca_master_data_{resource_id}.csv", index=False)
        print("Done!")
    else:
        print("No results found.")

if __name__ == "__main__":
    download_sample()
