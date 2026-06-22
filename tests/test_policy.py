import pytest
import os
import pandas as pd
from src.engine.model import IndianEconomyModel, FirmAgent
from src.engine.policy import PolicyIntervention, PolicyAnalyzer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_policy_shocks_application():
    """Verify that policy shocks are correctly scheduled and applied to the model."""
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    
    # Schedule a repo rate shock of +2% at tick 1
    shocks = [{'type': 'repo_rate_shock', 'value': 0.02, 'tick': 1}]
    model = IndianEconomyModel(data_path=data_path, policy_shocks=shocks, seed=42)
    
    initial_repo = model.repo_rate
    
    # Tick 0: Shock should not be applied yet
    model.step()
    assert model.repo_rate == initial_repo
    
    # Tick 1: Shock should be applied
    model.step()
    assert model.repo_rate == initial_repo + 0.02

def test_policy_analyzer():
    """Verify that PolicyAnalyzer executes baseline and counterfactual policy runs."""
    analyzer = PolicyAnalyzer()
    
    # Create simple intervention
    intervention = PolicyIntervention(
        name="Test Shock",
        description="Test repo rate shock",
        repo_rate_shock=0.01,
        shock_tick=0
    )
    
    # Run counterfactual with 2 ticks and fewer seeds for testing speed
    # We override config n_seeds to 1 for faster tests
    import yaml
    config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    old_n_seeds = config['run']['n_seeds']
    config['run']['n_seeds'] = 1
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
        
    try:
        scenario_data = analyzer.evaluate_intervention(intervention, ticks=2)
        assert isinstance(scenario_data, pd.DataFrame)
        assert len(scenario_data) == 2
    finally:
        # Restore configuration
        config['run']['n_seeds'] = old_n_seeds
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
