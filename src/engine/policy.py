import os
import pandas as pd
import yaml
from src.engine.run_simulation import run_simulation

# Determine repo root dir (3 levels up from this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(ROOT_DIR, "config", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

class PolicyIntervention:
    """Represents a specific macroeconomic policy shock to inject into the economy."""
    def __init__(self, name, description, shocks=None, repo_rate_shock=0.0, gst_shock=0.0, exchange_rate_shock=0.0, demonetisation_shock=0.0, carbon_price_shock=0.0, cbam_shock=0.0, shock_tick=0):
        self.name = name
        self.description = description
        if shocks is not None:
            self.shocks = shocks
        else:
            self.shocks = []
            if repo_rate_shock != 0.0:
                self.shocks.append({'type': 'repo_rate_shock', 'value': repo_rate_shock, 'tick': shock_tick})
            if gst_shock != 0.0:
                self.shocks.append({'type': 'gst_shock', 'value': gst_shock, 'tick': shock_tick})
            if exchange_rate_shock != 0.0:
                self.shocks.append({'type': 'exchange_rate_shock', 'value': exchange_rate_shock, 'tick': shock_tick})
            if demonetisation_shock != 0.0:
                self.shocks.append({'type': 'demonetisation_shock', 'value': demonetisation_shock, 'tick': shock_tick})
            if carbon_price_shock != 0.0:
                self.shocks.append({'type': 'carbon_price_shock', 'value': carbon_price_shock, 'tick': shock_tick})
            if cbam_shock != 0.0:
                self.shocks.append({'type': 'cbam_shock', 'value': cbam_shock, 'tick': shock_tick})

class PolicyAnalyzer:
    """Runs counterfactuals and evaluates the differential impact of policy interventions."""
    def __init__(self, baseline_csv_path=None):
        self.baseline_path = baseline_csv_path
        self.baseline_data = None
        if self.baseline_path and os.path.exists(self.baseline_path):
            self.baseline_data = pd.read_csv(self.baseline_path)
            
    def set_baseline(self, ticks=10):
        """Runs the un-shocked baseline simulation over multiple seeds and saves average results."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(root_dir, "data", "processed")
        self.baseline_path = os.path.join(out_dir, "baseline_results.csv")
        
        print("Running Baseline Simulation across multiple seeds...")
        n_seeds = config['run'].get('n_seeds', 10)
        master_seed = config['run'].get('master_seed', 42)
        
        runs = []
        for i in range(n_seeds):
            seed = master_seed + i
            print(f"Baseline Seed {seed} ({i+1}/{n_seeds})...")
            run_df = run_simulation(ticks=ticks, policy_shocks=None, seed=seed)
            runs.append(run_df)
            
        self.baseline_data = pd.concat(runs).groupby('Tick').mean()
        self.baseline_data.to_csv(self.baseline_path)
        print(f"Baseline average saved to {self.baseline_path}")
        
    def evaluate_intervention(self, intervention, ticks=10, save_path=None):
        """Runs the simulation under the intervention and compares it against baseline across multiple seeds."""
        if self.baseline_data is None:
            self.set_baseline(ticks=ticks)
            
        print(f"\n--- Evaluating Policy: {intervention.name} ---")
        print(f"Description: {intervention.description}")
        
        if not save_path:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out_dir = os.path.join(root_dir, "data", "processed")
            save_path = os.path.join(out_dir, "scenario_results.csv")
            
        n_seeds = config['run'].get('n_seeds', 10)
        master_seed = config['run'].get('master_seed', 42)
        
        runs = []
        for i in range(n_seeds):
            seed = master_seed + i
            print(f"Scenario Seed {seed} ({i+1}/{n_seeds})...")
            run_df = run_simulation(ticks=ticks, policy_shocks=intervention.shocks, seed=seed)
            runs.append(run_df)
            
        scenario_data = pd.concat(runs).groupby('Tick').mean()
        scenario_data.to_csv(save_path)
        print(f"Scenario average saved to {save_path}")
        
        # Compare final tick of average paths
        base_final = self.baseline_data.iloc[-1]
        scen_final = scenario_data.iloc[-1]
        
        output_diff = scen_final['Total_Output'] - base_final['Total_Output']
        output_pct = (output_diff / base_final['Total_Output']) * 100
        
        gini_diff = scen_final['Gini_Coefficient'] - base_final['Gini_Coefficient']
        
        print("\n=== Intervention Impact ===")
        print(f"Nominal Output Diff: {output_diff:+.2f} ({output_pct:+.2f}%)")
        print(f"Gini Coefficient Diff: {gini_diff:+.4f}")
        
        return scenario_data
