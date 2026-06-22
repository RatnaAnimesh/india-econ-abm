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

def test_carbon_pricing_and_cbam_shocks():
    """Verify that carbon price and CBAM shocks work as expected."""
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    
    # Run with a carbon price shock of 10.0 and CBAM shock of 0.20
    shocks = [
        {'type': 'carbon_price_shock', 'value': 10.0, 'tick': 0},
        {'type': 'cbam_shock', 'value': 0.20, 'tick': 0}
    ]
    model = IndianEconomyModel(data_path=data_path, policy_shocks=shocks, seed=42)
    
    # Verify values at initialization (before step)
    assert model.carbon_price == 0.0
    assert model.cbam_tariff == 0.0
    
    # Step once: shock is applied
    model.step()
    assert model.carbon_price == 10.0
    assert model.cbam_tariff == 0.20
    
    # Check that emissions and tax revenue are tracked
    df = model.datacollector.get_model_vars_dataframe()
    emissions = df.iloc[-1]['Total_Emissions']
    tax_rev = df.iloc[-1]['Carbon_Tax_Revenue']
    assert emissions >= 0.0
    assert tax_rev >= 0.0

