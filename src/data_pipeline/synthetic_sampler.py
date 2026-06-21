import os
import pandas as pd
import numpy as np
import yaml

# Determine repo root dir (3 levels up from this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(ROOT_DIR, "config", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

class SyntheticFirmGenerator:
    def __init__(self):
        self.num_agents = config['run']['n_agents']
        self.data_dir = os.path.join(ROOT_DIR, "data", "raw")
        self.out_dir = os.path.join(ROOT_DIR, "data", "processed")
        os.makedirs(self.out_dir, exist_ok=True)

    def _load_klems_data(self):
        """Loads and calculates macro sector average Capital and Labor shares from RBI KLEMS."""
        klems_path = os.path.join(self.data_dir, "INDIAKLEMS08072024.xlsx")
        
        # KSH_va = Capital Share in Value Added, LSH_va = Labor Share in Value Added
        df_k = pd.read_excel(klems_path, sheet_name="KSH_va", header=1)
        df_l = pd.read_excel(klems_path, sheet_name="LSH_va", header=1)
        
        # Merge on Industry
        df = pd.merge(
            df_k[['KLEMS Industry Description', '2022-23']].rename(columns={'2022-23': 'Cap_share'}),
            df_l[['KLEMS Industry Description', '2022-23']].rename(columns={'2022-23': 'Lab_share'}),
            on='KLEMS Industry Description'
        )
        
        # Classify into Macro Sectors
        def categorize_klems(desc):
            desc = str(desc).strip().lower()
            if "agriculture" in desc: return "Agriculture"
            if "mining" in desc or "manufacturing" in desc or "food" in desc or "textile" in desc or "wood" in desc or "paper" in desc or "coke" in desc or "chemical" in desc or "rubber" in desc or "mineral" in desc or "metal" in desc or "machinery" in desc or "electrical" in desc or "transport equipment" in desc or "electricity" in desc or "construction" in desc:
                return "Manufacturing"
            return "Services"
                
        df['Macro_Sector'] = df['KLEMS Industry Description'].apply(categorize_klems)
        
        # Compute mean share by macro sector
        sector_shares = df.groupby('Macro_Sector')[['Cap_share', 'Lab_share']].mean().to_dict('index')
        return sector_shares

    def _load_mca_state_weights(self):
        """Calculates State assignment probabilities from MCA data."""
        mca_path = os.path.join(self.data_dir, "mca_active_companies_2021.csv")
        df = pd.read_csv(mca_path)
        
        # Remove the 'Total' row
        df = df[df['State/UT'].str.lower() != 'total'].copy()
        
        # Clean columns: rename for easy access
        df.rename(columns={
            'State/UT': 'State', 
            'No. of Companies - Total': 'Total_Counts',
            'Authorized Capital - Total (In Crores)': 'Total_Capital'
        }, inplace=True)
        
        # Convert to numeric, handle commas if any
        df['Total_Counts'] = pd.to_numeric(df['Total_Counts'].astype(str).str.replace(',', ''), errors='coerce')
        df['Total_Capital'] = pd.to_numeric(df['Total_Capital'].astype(str).str.replace(',', ''), errors='coerce')
        df.dropna(subset=['Total_Counts', 'Total_Capital'], inplace=True)
        
        total_companies = df['Total_Counts'].sum()
        df['Prob'] = df['Total_Counts'] / total_companies
        
        return df

    def _load_mospi_sector_weights(self):
        """Calculates Sector assignment probabilities from MoSPI GDP data."""
        mospi_path = os.path.join(self.data_dir, "mospi_gdp_by_sector.csv")
        df = pd.read_csv(mospi_path)
        
        # Use Provisional Estimates for 2025-26
        df = df[(df['Revision'] == 'Provisional Estimates') & (df['Year'] == '2025-26')].copy()
        
        sector_mapping = {
            "Agriculture, Livestock, Forestry and Fishing": "Agriculture",
            "Mining and Quarrying": "Manufacturing",
            "Manufacturing": "Manufacturing",
            "Electricity, Gas, Water Supply & Other Utility Services": "Manufacturing",
            "Construction": "Manufacturing",
            "Trade, Hotels, Transport, Communication & Services Related to Broadcasting": "Services",
            "Financial, Real Estate & Professional Services": "Services",
            "Public Administration, Defence & Other Services": "Services"
        }
        
        df['Macro_Sector'] = df['Industry'].map(sector_mapping)
        df.dropna(subset=['Macro_Sector'], inplace=True) # Drops "Total Gross Value Added"
        
        df['Current Price'] = pd.to_numeric(df['Current Price'], errors='coerce')
        sector_gva = df.groupby('Macro_Sector')['Current Price'].sum()
        
        total_gva = sector_gva.sum()
        sector_probs = sector_gva / total_gva
        return sector_probs.to_dict()

    def generate(self):
        print(f"Generating {self.num_agents} synthetic agents...")
        
        # 1. Load Distributions
        mca_df = self._load_mca_state_weights()
        sector_probs = self._load_mospi_sector_weights()
        klems_shares = self._load_klems_data()
        
        print("Real KLEMS Aggregates Mapped:", klems_shares)
        
        # 2. Sample States
        states = np.random.choice(
            mca_df['State'].values, 
            size=self.num_agents, 
            p=mca_df['Prob'].values
        )
        
        # 3. Sample Sectors
        sectors = np.random.choice(
            list(sector_probs.keys()), 
            size=self.num_agents, 
            p=list(sector_probs.values())
        )
        
        # 4. Generate Base DataFrame
        firms = pd.DataFrame({
            "CIN": [f"U{np.random.randint(10000, 99999)}{state[:2].upper()}2024PTC{np.random.randint(100000, 999999)}" for state in states],
            "State": states,
            "Sector": sectors
        })
        
        # 5. Assign Capital (Zipf's Law Distribution, S^-gamma, gamma approx 2)
        # We use a Pareto distribution which is the continuous equivalent of Zipf.
        # Pareto alpha parameter = gamma - 1. So for gamma=2, alpha=1.
        state_avg_cap = dict(zip(mca_df['State'], mca_df['Total_Capital'] / mca_df['Total_Counts']))
        
        def sample_capital_zipf(state):
            # We anchor the min value (xm) based on the state average to keep realistic scales
            # Mean of Pareto(xm, alpha) = xm * alpha / (alpha - 1) for alpha > 1
            # For alpha=1.1, Mean = xm * 11 -> xm = Mean / 11
            alpha = 1.1 
            mu = state_avg_cap[state]
            xm = max(0.01, mu / 11.0)
            return np.random.pareto(alpha) * xm + xm
            
        firms['Capital'] = firms['State'].apply(sample_capital_zipf)
        
        # 6. Assign Real KLEMS Cobb-Douglas parameters
        firms['Cap_share'] = firms['Sector'].map(lambda s: klems_shares[s]['Cap_share'])
        firms['Lab_share'] = firms['Sector'].map(lambda s: klems_shares[s]['Lab_share'])
        
        # 7. Initialize baseline Labor and Productivity
        # Map generic labor scaling
        capital_to_labor_scaling = config['initialization']['capital_to_labor_scaling']
        firms['Labor'] = (firms['Capital'] * capital_to_labor_scaling).astype(int)
        firms['Labor'] = firms['Labor'].clip(lower=1)
        
        # Add Debt based on empirical average Debt-to-Equity (Capital) ratio of ~0.8
        # We add some random lognormal noise so not every firm has exactly 0.8 leverage
        leverage_ratios = np.random.lognormal(mean=np.log(0.8), sigma=0.5, size=self.num_agents)
        firms['Debt'] = firms['Capital'] * leverage_ratios
        
        firms['Productivity'] = 1.0 # Base TFP
        
        # Save output
        out_path = os.path.join(self.out_dir, "synthetic_firms.csv")
        firms.to_csv(out_path, index=False)
        print(f"Successfully generated {len(firms)} agents to {out_path}.")
        
        # Print summary statistics
        print("\n=== Agent Generation Summary ===")
        print(f"Total Agents: {len(firms)}")
        print("\nSector Distribution:")
        print(firms['Sector'].value_counts(normalize=True))
        print("\nCapital Distribution (Crores):")
        print(firms['Capital'].describe())
        print("\nKLEMS Parameters (Averages):")
        print(firms.groupby('Sector')[['Cap_share', 'Lab_share']].mean())

if __name__ == "__main__":
    generator = SyntheticFirmGenerator()
    generator.generate()
