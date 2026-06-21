import os
import argparse
import pandas as pd
from src.engine.model import IndianEconomyModel

def run_simulation(ticks=10, policy_shocks=None, save_path=None):
    print(f"Initializing Indian Economy ABM for {ticks} ticks (years)...")
    if policy_shocks:
        print(f"Applying Policy Shocks: {policy_shocks}")
    
    # Use relative pathing from root
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_dir, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, policy_shocks=policy_shocks)
    
    # Run the simulation loop
    for i in range(ticks):
        print(f"Running tick {i+1}/{ticks}...")
        model.step()
        
    print("Simulation complete. Extracting macro data...")
    
    # Extract data collector output
    results_df = model.datacollector.get_model_vars_dataframe()
    results_df.index.name = "Tick"
    
    # Flatten the State_Output dictionary into separate columns
    if 'State_Output' in results_df.columns:
        state_df = results_df['State_Output'].apply(pd.Series)
        # Prefix columns with 'State_' to avoid name collisions
        state_df = state_df.add_prefix('State_')
        results_df = pd.concat([results_df.drop('State_Output', axis=1), state_df], axis=1)
        
    # Scale aggregated volume metrics by 100 (since 15k agents represents 1.5M firms)
    scale_factor = 100.0
    non_volume_metrics = ['Price_Level', 'Gini_Coefficient', 'Agri_Price_Multiplier', 'Mfg_Price_Multiplier', 'Svc_Price_Multiplier']
    for col in results_df.columns:
        if col not in non_volume_metrics:
            results_df[col] = results_df[col] * scale_factor
            
    # Save results
    if not save_path:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(root_dir, "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "simulation_results.csv")
    
    results_df.to_csv(save_path)
    print(f"Results saved to {save_path}")
    print("\n=== Final Macroeconomic State ===")
    print(results_df.tail(1).T)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Indian Economy ABM.")
    parser.add_argument("--repo_shock", type=float, default=0.0, help="Additive shock to RBI Repo Rate (e.g. 0.02 for +2%)")
    parser.add_argument("--gst_shock", type=float, default=0.0, help="Additive shock to GST Rates (e.g. -0.05 for -5%)")
    parser.add_argument("--exchange_shock", type=float, default=0.0, help="Additive shock to Exchange Rate (e.g. 0.1 for 10% depreciation)")
    args = parser.parse_args()
    
    policy_shocks = {
        'repo_rate_shock': args.repo_shock,
        'gst_shock': args.gst_shock,
        'exchange_rate_shock': args.exchange_shock
    }
    
    run_simulation(policy_shocks=policy_shocks)
