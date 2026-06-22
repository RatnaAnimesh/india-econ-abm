import pytest
import os
from src.engine.model import IndianEconomyModel, FirmAgent

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_stock_flow_consistency_equations():
    """
    Rigorously verify the balance sheet identities of institutional sectors.
    At every step, Assets must equal Liabilities + Equity (Net Worth).
    """
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, seed=42)
    
    # Verify Day Zero stock balances
    bank = model.commercial_bank
    firm_debt_sum = sum(f.debt for f in model.active_firms)
    firm_deposits_sum = sum(f.deposits for f in model.active_firms)
    hh_deposits_sum = sum(h.deposits for h in model.households)
    
    # Check bank loan assets vs firm liabilities (debt)
    # The bank loan asset ledger must match the sum of firm debt
    assert abs(bank.loans - firm_debt_sum) < 1e-5
    
    # Check bank deposit liabilities vs firm + household asset deposits
    total_private_deposits = firm_deposits_sum + hh_deposits_sum
    assert abs(bank.deposits - total_private_deposits) < 1e-5
    
    # Check balance sheet identity (Equity = Assets - Liabilities)
    # net_worth property enforces this by construction, but let's check values
    equity = bank.net_worth
    assets = bank.loans + bank.reserves
    liabilities = bank.deposits
    assert abs(equity - (assets - liabilities)) < 1e-5
    
    # Run the model for 2 steps and verify balance sheet holds at each step
    for tick in range(2):
        model.step()
        
        # Recalculate sums
        firm_debt_sum = sum(f.debt for f in model.active_firms)
        firm_deposits_sum = sum(f.deposits for f in model.active_firms)
        hh_deposits_sum = sum(h.deposits for h in model.households)
        total_private_deposits = firm_deposits_sum + hh_deposits_sum
        
        # Verify SFC Loans
        assert abs(bank.loans - firm_debt_sum) < 1e-5
        
        # Verify SFC Deposits
        assert abs(bank.deposits - total_private_deposits) < 1e-5
        
        # Verify SFC Balance Sheet Identity
        assert abs(bank.net_worth - (bank.loans + bank.reserves - bank.deposits)) < 1e-5
        
        # Verify Gini is in valid bounds
        gini = model.datacollector.get_model_vars_dataframe().iloc[-1]['Gini_Coefficient']
        assert 0.0 <= gini <= 1.0
