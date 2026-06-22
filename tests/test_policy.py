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
    
    # Override policy config in memory without writing to disk
    import src.engine.policy
    old_n_seeds = src.engine.policy.config['run'].get('n_seeds', 10)
    src.engine.policy.config['run']['n_seeds'] = 1
    
    try:
        scenario_data = analyzer.evaluate_intervention(intervention, ticks=2)
        assert isinstance(scenario_data, pd.DataFrame)
        assert len(scenario_data) == 2
    finally:
        # Restore configuration
        src.engine.policy.config['run']['n_seeds'] = old_n_seeds
