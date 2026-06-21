import mesa
import yaml
import os
import random
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

def compute_gini(model):
    """Compute the Gini coefficient of capital distribution."""
    agent_wealths = [agent.capital for agent in model.firm_agents]
    x = sorted(agent_wealths)
    N = len(agent_wealths)
    if N == 0 or sum(x) == 0:
        return 0
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return (1 + (1/N) - 2*B)

class FirmAgent:
    """A firm in the Indian economy."""
    def __init__(self, unique_id, sector, state, capital, labor, cap_share, lab_share, debt, model, shocks=None):
        self.unique_id = unique_id
        self.sector = sector
        self.state = state
        self.capital = capital
        self.labor = labor
        self.cap_share = cap_share
        self.lab_share = lab_share
        self.debt = debt
        self.model = model
        self.shocks = shocks or {}
        
        # Load empirical parameters from config
        self.wage_rate = config['agent_logic']['baseline_wage_rate']
        self.depreciation_rate = config['agent_logic']['depreciation_rate']
        self.investment_rate = config['agent_logic']['investment_rate']
        self.profit_labor_growth_rate = config['agent_logic']['profit_labor_growth_rate']
        self.loss_labor_shrink_rate = config['agent_logic']['loss_labor_shrink_rate']
        self.tfp_growth_rate = config['agent_logic']['tfp_growth_rates'][self.sector] / 100.0
        
        # Fiscal Policy (with potential shocks)
        self.corporate_tax_rate = config['fiscal_policy']['corporate_tax_rate']
        base_gst = config['fiscal_policy']['gst_rates'][self.sector]
        # Shock: additive or multiplier. Let's make it additive (e.g. -0.05 for a 5% GST cut)
        gst_shock = self.shocks.get('gst_shock', 0.0)
        self.gst_rate = max(0.0, base_gst + gst_shock)
        
        # Monetary Policy (with potential shocks)
        base_repo = config['monetary_policy']['rbi_repo_rate']
        repo_shock = self.shocks.get('repo_rate_shock', 0.0) # e.g. +0.02 for a 200 bps hike
        self.interest_rate = max(0.0, base_repo + repo_shock + config['monetary_policy']['corporate_spread'])
        
        # Trade Policy
        self.export_propensity = config['trade_policy']['export_propensity'][self.sector]
        
        # State variables for tracking
        self.output = 0
        self.profit = 0
        self.tfp = 1.0 # Total Factor Productivity (A)
        self.bankruptcies = 0

    def step(self):
        # 1. Produce Output (Cobb-Douglas: Y = A * K^alpha * L^beta)
        real_output = self.tfp * (self.capital ** self.cap_share) * (self.labor ** self.lab_share)
        
        # 2. Nominal Conversion & Trade
        price_level = self.model.price_level
        # Shock: exchange rate shock. e.g. +0.10 means 10% depreciation
        base_exchange_shock = config['trade_policy']['exchange_rate_shock']
        exchange_shock = base_exchange_shock + self.shocks.get('exchange_rate_shock', 0.0)
        
        # Exports benefit from depreciation (shock > 1)
        trade_multiplier = 1.0 + (self.export_propensity * (exchange_shock - 1.0))
        nominal_revenue = real_output * price_level * trade_multiplier
        self.output = nominal_revenue  # Track nominal revenue
        
        # 3. Pay Wages (Indexed to inflation)
        wage_bill = self.wage_rate * price_level * self.labor
        
        # 4. Pay Interest on Debt
        interest_payment = self.debt * self.interest_rate
        
        # 5. Depreciation (Nominal cost of capital replacement)
        depreciation_cost = self.capital * self.depreciation_rate * price_level
        
        # 6. Taxes (GST)
        gst_payment = nominal_revenue * self.gst_rate
        self.model.total_tax_collection += gst_payment
        
        # 7. Calculate EBT (Earnings Before Tax)
        ebt = nominal_revenue - wage_bill - depreciation_cost - interest_payment - gst_payment
        
        # 8. Corporate Tax & Net Profit
        if ebt > 0:
            corporate_tax = ebt * self.corporate_tax_rate
            self.model.total_tax_collection += corporate_tax
            self.profit = ebt - corporate_tax
        else:
            self.profit = ebt
            
        # 9. Update Capital, Labor, and Debt based on Profitability
        if self.profit > 0:
            # Reinvest profit into new capital
            real_investment = (self.profit * self.investment_rate) / price_level
            self.capital += real_investment
            
            # Optionally pay down some debt with remaining profit
            debt_paydown = (self.profit * (1 - self.investment_rate)) * 0.1
            self.debt = max(0, self.debt - debt_paydown)
            
            # Hire workers
            self.labor = int(self.labor * self.profit_labor_growth_rate)
        else:
            # Loss scenario
            self.capital -= (depreciation_cost / price_level)
            self.labor = int(self.labor * self.loss_labor_shrink_rate)
            # Add loss to debt (borrowing to survive)
            self.debt += abs(self.profit)
            
            # Bankruptcy Check
            if self.debt > (self.capital * price_level * 2.0):
                self.bankruptcies += 1
                self.debt = self.debt * 0.1 # Debt restructuring
                self.capital = max(0.01, self.capital * 0.5) # Sell off assets
                self.labor = max(1, int(self.labor * 0.5)) # Massive layoffs
                
        # Prevent zero or negative capital/labor
        self.capital = max(0.01, self.capital)
        self.labor = max(1.0, self.labor)
        
        # 10. Technological Progress (TFP Growth)
        self.tfp *= (1.0 + self.tfp_growth_rate)


class IndianEconomyModel(mesa.Model):
    """The macro-economy model containing all firms and government."""
    def __init__(self, data_path=None, policy_shocks=None):
        super().__init__()
        self.policy_shocks = policy_shocks or {}
        
        import pandas as pd
        if data_path:
            synthetic_firms_df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide data_path to synthetic firms.")
            
        self.num_agents = len(synthetic_firms_df)
        self.firm_agents = []
        
        # Macro variables
        self.price_level = 1.0
        self.inflation_rate = config['monetary_policy']['target_inflation_rate']
        self.total_tax_collection = 0.0
        
        # Create agents
        for idx, row in synthetic_firms_df.iterrows():
            debt = row.get("Debt", 0.0)
            agent = FirmAgent(
                unique_id=row['CIN'],
                sector=row['Sector'],
                state=row['State'],
                capital=row['Capital'],
                labor=row['Labor'],
                cap_share=row['Cap_share'],
                lab_share=row['Lab_share'],
                debt=debt,
                model=self,
                shocks=self.policy_shocks
            )
            self.firm_agents.append(agent)
            
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total_Output": lambda m: sum([a.output for a in m.firm_agents]),
                "Total_Profit": lambda m: sum([a.profit for a in m.firm_agents]),
                "Total_Capital": lambda m: sum([a.capital for a in m.firm_agents]),
                "Total_Labor": lambda m: sum([a.labor for a in m.firm_agents]),
                "Total_Debt": lambda m: sum([a.debt for a in m.firm_agents]),
                "Total_Tax_Revenue": lambda m: m.total_tax_collection,
                "Price_Level": lambda m: m.price_level,
                "Bankruptcies": lambda m: sum([a.bankruptcies for a in m.firm_agents]),
                "Gini_Coefficient": compute_gini,
                "Agri_Output": lambda m: sum([a.output for a in m.firm_agents if a.sector == "Agriculture"]),
                "Mfg_Output": lambda m: sum([a.output for a in m.firm_agents if a.sector == "Manufacturing"]),
                "Svc_Output": lambda m: sum([a.output for a in m.firm_agents if a.sector == "Services"]),
                "Agri_Profit": lambda m: sum([a.profit for a in m.firm_agents if a.sector == "Agriculture"]),
                "Mfg_Profit": lambda m: sum([a.profit for a in m.firm_agents if a.sector == "Manufacturing"]),
                "Svc_Profit": lambda m: sum([a.profit for a in m.firm_agents if a.sector == "Services"])
            }
        )

    def step(self):
        """Advance the model by one step (one year)."""
        # Inflate the price level
        self.price_level *= (1.0 + self.inflation_rate)
        
        # Reset tax collection for the year
        self.total_tax_collection = 0.0
        
        # Randomize execution order
        random.shuffle(self.firm_agents)
        for agent in self.firm_agents:
            agent.step()
            
        self.datacollector.collect(self)
