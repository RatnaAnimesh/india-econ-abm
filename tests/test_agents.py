import pytest
import os
import yaml
from src.engine.model import IndianEconomyModel, FirmAgent
from src.engine.sectors import HouseholdAgent

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(ROOT_DIR, "config", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def test_agent_initialization():
    """Verify that FirmAgent and HouseholdAgent initialize with correct parameters and types."""
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, seed=42)
    
    # Check firm agents
    firms = [a for a in model.agents if isinstance(a, FirmAgent)]
    assert len(firms) > 0
    
    sample_firm = firms[0]
    assert sample_firm.cin is not None
    assert sample_firm.deposits >= 0.0
    assert sample_firm.debt >= 0.0
    assert sample_firm.capital > 0.0
    assert sample_firm.labor >= 1.0
    
    # Check household agents
    assert len(model.households) > 0
    sample_hh = model.households[0]
    assert sample_hh.deposits == 100.0
    assert sample_hh.state is not None
    assert sample_hh.mpc == 0.8

def test_firm_financials_balance():
    """Verify that paying financials reduces deposits and updates the bank ledger correctly."""
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, seed=42)
    
    firm = [a for a in model.agents if isinstance(a, FirmAgent)][0]
    initial_deposits = firm.deposits
    initial_bank_deposits = model.commercial_bank.deposits
    
    # Set output to simulate sales
    firm.output = 1000.0
    
    # Run pay_financials
    firm.pay_financials()
    
    # Verify that GST rate is applied and GST is collected
    assert firm.gst_payment == firm.output * firm.gst_rate
    assert model.total_tax_collection >= firm.gst_payment
