import os
import pandas as pd
from src.engine.run_simulation import run_simulation

class PolicyIntervention:
    """Represents a specific macroeconomic policy shock to inject into the economy."""
    def __init__(self, name, description, repo_rate_shock=0.0, gst_shock=0.0, exchange_rate_shock=0.0, demonetisation_shock=0.0):
        self.name = name
        self.description = description
        self.shocks = {
            'repo_rate_shock': repo_rate_shock,
            'gst_shock': gst_shock,
            'exchange_rate_shock': exchange_rate_shock,
            'demonetisation_shock': demonetisation_shock
        }

class PolicyAnalyzer:
    """Runs counterfactuals and evaluates the differential impact of policy interventions."""
    def __init__(self, baseline_csv_path=None):
        self.baseline_path = baseline_csv_path
        self.baseline_data = None
        if self.baseline_path and os.path.exists(self.baseline_path):
            self.baseline_data = pd.read_csv(self.baseline_path)
            
    def set_baseline(self, ticks=10):
        """Runs the un-shocked baseline simulation and saves it."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(root_dir, "data", "processed")
        self.baseline_path = os.path.join(out_dir, "baseline_results.csv")
        
        print("Running Baseline Simulation...")
        # Note: run_simulation handles its own scaling
        run_simulation(ticks=ticks, policy_shocks=None, save_path=self.baseline_path)
        self.baseline_data = pd.read_csv(self.baseline_path)
        print(f"Baseline saved to {self.baseline_path}")
        
    def evaluate_intervention(self, intervention, ticks=10, save_path=None):
        """Runs the simulation under the intervention and compares it against baseline."""
        if self.baseline_data is None:
            self.set_baseline(ticks=ticks)
            
        print(f"\n--- Evaluating Policy: {intervention.name} ---")
        print(f"Description: {intervention.description}")
        
        if not save_path:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out_dir = os.path.join(root_dir, "data", "processed")
            save_path = os.path.join(out_dir, "scenario_results.csv")
            
        run_simulation(ticks=ticks, policy_shocks=intervention.shocks, save_path=save_path)
        scenario_data = pd.read_csv(save_path)
        
        # Compare final tick
        base_final = self.baseline_data.iloc[-1]
        scen_final = scenario_data.iloc[-1]
        
        output_diff = scen_final['Total_Output'] - base_final['Total_Output']
        output_pct = (output_diff / base_final['Total_Output']) * 100
        
        gini_diff = scen_final['Gini_Coefficient'] - base_final['Gini_Coefficient']
        
        print("\n=== Intervention Impact ===")
        print(f"Nominal Output Diff: {output_diff:,.0f} ({output_pct:+.2f}%)")
        print(f"Gini Coefficient Diff: {gini_diff:+.4f}")
        
        return scenario_data
