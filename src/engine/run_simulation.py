import os
import argparse
import yaml
import pandas as pd
from src.engine.model import IndianEconomyModel

# Determine repo root dir (3 levels up from this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(ROOT_DIR, "config", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def run_simulation(ticks=10, policy_shocks=None, save_path=None, seed=None):
    print(f"Initializing Indian Economy ABM for {ticks} ticks (years)...")
    if policy_shocks:
        print(f"Applying Policy Shocks: {policy_shocks}")
        
    if seed is None:
        seed = config['run'].get('master_seed', 42)
    
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, policy_shocks=policy_shocks, seed=seed)
    
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
        state_df = state_df.add_prefix('State_')
        results_df = pd.concat([results_df.drop('State_Output', axis=1), state_df], axis=1)
        
    # Scale aggregated volume metrics by representing the full Indian economy
    # total active firms in MoSPI/MCA data approx 1.5 million.
    n_agents = config['run']['n_agents']
    scale_factor = 1500000.0 / n_agents
    
    non_volume_metrics = ['Price_Level', 'Gini_Coefficient', 'Agri_Price_Multiplier', 'Mfg_Price_Multiplier', 'Svc_Price_Multiplier']
    for col in results_df.columns:
        if col not in non_volume_metrics and not col.startswith('State_'):
            results_df[col] = results_df[col] * scale_factor
            
    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path)
        print(f"Results saved to {save_path}")
        
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Indian Economy ABM.")
    parser.add_argument("--ticks", type=int, default=10, help="Number of ticks (years)")
    parser.add_argument("--repo_shock", type=float, default=0.0, help="Additive shock to RBI Repo Rate (e.g. 0.02 for +2%)")
    parser.add_argument("--gst_shock", type=float, default=0.0, help="Additive shock to GST Rates (e.g. -0.05 for -5%)")
    parser.add_argument("--exchange_shock", type=float, default=0.0, help="Additive shock to Exchange Rate (e.g. 0.1 for 10% depreciation)")
    args = parser.parse_args()
    
    # Pack shocks as a policy_shocks list
    shocks = []
    if args.repo_shock != 0.0:
        shocks.append({'type': 'repo_rate_shock', 'value': args.repo_shock, 'tick': 0})
    if args.gst_shock != 0.0:
        shocks.append({'type': 'gst_shock', 'value': args.gst_shock, 'tick': 0})
    if args.exchange_shock != 0.0:
        shocks.append({'type': 'exchange_rate_shock', 'value': args.exchange_shock, 'tick': 0})
        
    run_simulation(ticks=args.ticks, policy_shocks=shocks)
