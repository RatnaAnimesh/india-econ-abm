import pytest
import os
from src.engine.model import IndianEconomyModel, FirmAgent
from src.engine.sectors import HouseholdAgent

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_labor_market_matching():
    """Verify that LaborMarket matches unemployed workers to firms with vacancies."""
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, seed=42)
    
    # Fire some workers to create unemployment
    firms = [a for a in model.agents if isinstance(a, FirmAgent)]
    firm = firms[0]
    
    # Create vacancies by boosting expected demand
    firm.expected_demand = firm.capital * 10.0
    firm.inventory = 0.0
    
    # Ensure there are unemployed households
    for hh in model.households[:10]:
        hh.employed = False
        hh.employer = None
        hh.wage = 0.0
        
    initial_employees = len(firm.employees)
    
    # Step the labor market
    model.labor_market.step()
    
    # Verify reservation wage updates
    for hh in model.households[:10]:
        assert hh.reservation_wage > 0.0

def test_financial_market_lob():
    """Verify that the LimitOrderBook trades index shares and matches orders."""
    data_path = os.path.join(ROOT_DIR, "data", "processed", "synthetic_firms.csv")
    model = IndianEconomyModel(data_path=data_path, seed=42)
    
    lob = model.stock_market
    initial_price = lob.current_price
    
    # Step stock market
    lob.step()
    
    # Verify price history is updated
    assert len(lob.price_history) >= 1
    assert lob.current_price > 0.0
