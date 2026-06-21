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
        gst_shock = self.shocks.get('gst_shock', 0.0)
        self.gst_rate = max(0.0, base_gst + gst_shock)
        
        # Monetary Policy (with potential shocks)
        base_repo = config['monetary_policy']['rbi_repo_rate']
        repo_shock = self.shocks.get('repo_rate_shock', 0.0)
        self.interest_rate = max(0.0, base_repo + repo_shock + config['monetary_policy']['corporate_spread'])
        
        # Trade Policy
        self.export_propensity = config['trade_policy']['export_propensity'][self.sector]
        
        # IO Matrix Requirements
        self.io_reqs = {
            "Agriculture": config['io_matrix']['Agriculture'][self.sector],
            "Manufacturing": config['io_matrix']['Manufacturing'][self.sector],
            "Services": config['io_matrix']['Services'][self.sector]
        }
        
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
        base_exchange_shock = config['trade_policy']['exchange_rate_shock']
        exchange_shock = base_exchange_shock + self.shocks.get('exchange_rate_shock', 0.0)
        
        trade_multiplier = 1.0 + (self.export_propensity * (exchange_shock - 1.0))
        nominal_revenue = real_output * price_level * trade_multiplier
        self.output = nominal_revenue  
        
        # 3. Supply Chain Intermediate Consumption
        # If supply of an input drops, the cost to procure it spikes
        intermediate_cost = 0.0
        for input_sec, req_pct in self.io_reqs.items():
            base_cost = nominal_revenue * req_pct
            supply_multiplier = self.model.supply_multipliers.get(input_sec, 1.0)
            # Prevent division by zero, max price spike is 5x
            scarcity_price_modifier = min(5.0, 1.0 / max(0.2, supply_multiplier))
            intermediate_cost += (base_cost * scarcity_price_modifier)
        
        # 4. Pay Wages (Indexed to inflation)
        wage_bill = self.wage_rate * price_level * self.labor
        
        # 5. Pay Interest on Debt
        interest_payment = self.debt * self.interest_rate
        
        # 6. Depreciation
        depreciation_cost = self.capital * self.depreciation_rate * price_level
        
        # 7. Taxes (GST)
        gst_payment = nominal_revenue * self.gst_rate
        self.model.total_tax_collection += gst_payment
        
        # 8. Calculate EBT (Earnings Before Tax)
        ebt = nominal_revenue - intermediate_cost - wage_bill - depreciation_cost - interest_payment - gst_payment
        
        # 9. Corporate Tax & Net Profit
        if ebt > 0:
            corporate_tax = ebt * self.corporate_tax_rate
            self.model.total_tax_collection += corporate_tax
            self.profit = ebt - corporate_tax
        else:
            self.profit = ebt
            
        # 10. Update Capital, Labor, and Debt based on Profitability
        if self.profit > 0:
            real_investment = (self.profit * self.investment_rate) / price_level
            self.capital += real_investment
            
            debt_paydown = (self.profit * (1 - self.investment_rate)) * 0.1
            self.debt = max(0, self.debt - debt_paydown)
            
            self.labor = int(self.labor * self.profit_labor_growth_rate)
        else:
            self.capital -= (depreciation_cost / price_level)
            self.labor = int(self.labor * self.loss_labor_shrink_rate)
            self.debt += abs(self.profit)
            
            # Bankruptcy Check
            if self.debt > (self.capital * price_level * 2.0):
                self.bankruptcies += 1
                self.debt = self.debt * 0.1
                self.capital = max(0.01, self.capital * 0.5)
                self.labor = max(1, int(self.labor * 0.5))
                
        self.capital = max(0.01, self.capital)
        self.labor = max(1.0, self.labor)
        self.tfp *= (1.0 + self.tfp_growth_rate)


def compute_state_output(model):
    """Aggregate nominal output by state."""
    state_totals = {}
    for agent in model.firm_agents:
        state_totals[agent.state] = state_totals.get(agent.state, 0.0) + agent.output
    return state_totals

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
        
        # IO Supply Tracking
        self.supply_multipliers = {
            "Agriculture": 1.0,
            "Manufacturing": 1.0,
            "Services": 1.0
        }
        self.previous_supply = None
        self.baseline_supply = None
        
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
                "Svc_Profit": lambda m: sum([a.profit for a in m.firm_agents if a.sector == "Services"]),
                "State_Output": compute_state_output,
                "Agri_Price_Multiplier": lambda m: 1.0 / max(0.2, m.supply_multipliers.get("Agriculture", 1.0)),
                "Mfg_Price_Multiplier": lambda m: 1.0 / max(0.2, m.supply_multipliers.get("Manufacturing", 1.0)),
                "Svc_Price_Multiplier": lambda m: 1.0 / max(0.2, m.supply_multipliers.get("Services", 1.0))
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
            
        # Calculate sector supply for the next tick's intermediate consumption
        current_supply = {
            "Agriculture": sum([a.output for a in self.firm_agents if a.sector == "Agriculture"]),
            "Manufacturing": sum([a.output for a in self.firm_agents if a.sector == "Manufacturing"]),
            "Services": sum([a.output for a in self.firm_agents if a.sector == "Services"])
        }
        
        if self.baseline_supply is None:
            self.baseline_supply = current_supply
        else:
            # Update supply multipliers based on deviation from baseline
            for sec in current_supply:
                if self.baseline_supply[sec] > 0:
                    self.supply_multipliers[sec] = current_supply[sec] / self.baseline_supply[sec]
                    
        self.previous_supply = current_supply
            
        self.datacollector.collect(self)
