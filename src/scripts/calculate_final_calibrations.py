import os
import pandas as pd
import numpy as np
import yaml

# 1. Calculate Empirical Sigma for Capital Inequality
mca_path = "data/raw/mca_active_companies_2021.csv"
mca_df = pd.read_csv(mca_path)
mca_df = mca_df[mca_df['State/UT'].str.lower() != 'total'].copy()

mca_df.rename(columns={
    'No. of Companies - Total': 'Total_Counts',
    'Authorized Capital - Total (In Crores)': 'Total_Capital'
}, inplace=True)

mca_df['Total_Counts'] = pd.to_numeric(mca_df['Total_Counts'].astype(str).str.replace(',', ''), errors='coerce')
mca_df['Total_Capital'] = pd.to_numeric(mca_df['Total_Capital'].astype(str).str.replace(',', ''), errors='coerce')
mca_df.dropna(subset=['Total_Counts', 'Total_Capital'], inplace=True)

# Calculate average capital per firm in each state
avg_capital_per_state = mca_df['Total_Capital'] / mca_df['Total_Counts']

# To find lognormal sigma, we calculate the standard deviation of the log of the averages
log_capitals = np.log(avg_capital_per_state)
empirical_sigma = log_capitals.std()

# 2. Calculate TFP Growth averages
klems_path = "data/raw/INDIAKLEMS08072024.xlsx"
df_tfp = pd.read_excel(klems_path, sheet_name="TFPG_va", header=1)

def categorize(desc):
    desc = str(desc).lower()
    if "agriculture" in desc:
        return "Agriculture"
    elif any(s in desc for s in ["trade", "hotel", "transport", "financial", "business", "public", "education", "health", "post", "storage"]):
        return "Services"
    else:
        return "Manufacturing"

# Map to our 3 macro sectors
df_tfp['Macro_Sector'] = df_tfp['KLEMS Industry Description'].apply(categorize)

# We want the average TFP growth over the last 10 years (e.g., 2012-13 to 2022-23)
years = ['2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
# Average growth across those years for each row
df_tfp['Avg_10yr_TFPG'] = df_tfp[years].mean(axis=1)

# Average across sectors
tfp_growth_by_sector = df_tfp.groupby('Macro_Sector')['Avg_10yr_TFPG'].mean().to_dict()

# 3. Update config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Update agents to 15,000 (1:100 scale ratio of ~1.5m firms)
config["run"]["n_agents"] = 15000

# Update sigma
config["initialization"]["lognormal_sigma_default"] = float(empirical_sigma)

# Update TFP
if "agent_logic" not in config:
    config["agent_logic"] = {}
config["agent_logic"]["tfp_growth_rates"] = {
    "Agriculture": float(tfp_growth_by_sector["Agriculture"]),
    "Manufacturing": float(tfp_growth_by_sector["Manufacturing"]),
    "Services": float(tfp_growth_by_sector["Services"])
}

with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Empirical Sigma computed: {empirical_sigma}")
print(f"Empirical TFP Growth (Percentages) computed: {tfp_growth_by_sector}")
print("Config updated successfully.")
