from mesa import Model, Agent, DataCollector
import yaml
import sys
import os
import random
import numpy as np

# Go up two levels to root, then into config
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

config_path = os.path.join(root_dir, "config", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

def compute_gini(model):
    """Compute the Gini coefficient of capital distribution."""
    agent_wealths = [a.capital for a in model.agents if hasattr(a, 'capital')]
    x = sorted(agent_wealths)
    N = len(agent_wealths)
    if N == 0 or sum(x) == 0:
        return 0
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return (1 + (1/N) - 2*B)

class FirmAgent(Agent):
    """A firm in the Indian economy."""
    def __init__(self, model, unique_id, sector, state, capital, labor, cap_share, lab_share, debt, shocks=None):
        super().__init__(model)
        self.cin = unique_id  # Store the CIN since Mesa 3 generates its own unique_id
        self.sector = sector
        self.state = state
        self.capital = capital
        self.labor = labor
        self.cap_share = cap_share
        self.lab_share = lab_share
        self.debt = debt
        self.deposits = debt * 0.1 # initial liquidity buffer
        self.net_worth = self.capital + self.deposits - self.debt
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
        
        # Fetch real-time macro parameters via DataLoader
        from src.data_pipeline.data_loader import DataLoader
        loader = DataLoader(config_path=config_path)
        macro_data = loader.fetch_macro_data(allow_stale=True)
        
        base_repo = config['monetary_policy']['rbi_repo_rate']
        if "repo_rate" in macro_data and "value" in macro_data["repo_rate"]:
            base_repo = macro_data["repo_rate"]["value"]
            
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
        
        # Adaptive Expectations & Production
        self.expected_demand = 0.0
        self.inventory = 0.0
        self.learning_rate = 0.2 # eta
        self.buffer_ratio = 0.1 # nu
        
    def step(self):
        # 0. Adaptive Expectations
        # Update expected demand based on previous sales (output)
        if self.expected_demand == 0.0:
            self.expected_demand = self.capital * 0.5 # initial heuristic
            
        sales_last_period = self.output / max(1.0, self.model.price_level) # Real sales
        self.expected_demand = self.expected_demand + self.learning_rate * (sales_last_period - self.expected_demand)
        
        target_production = max(0.0, self.expected_demand * (1.0 + self.buffer_ratio) - self.inventory)

        # 1. Produce Output (Cobb-Douglas Capacity limit)
        max_capacity = self.tfp * (self.capital ** self.cap_share) * (self.labor ** self.lab_share)
        real_output = min(target_production, max_capacity)
        
        # Update inventory
        self.inventory += real_output
        
        # 2. Nominal Conversion & Trade
        price_level = self.model.price_level
        base_exchange_shock = config['trade_policy']['exchange_rate_shock']
        exchange_shock = base_exchange_shock + self.shocks.get('exchange_rate_shock', 0.0)
        
        trade_multiplier = 1.0 + (self.export_propensity * (exchange_shock - 1.0))
        
        # Assume all produced output + inventory is offered to the market.
        # In a full SFC model, actual sales depend on household consumption matching.
        # Here we approximate sales as market clearing up to expected demand + random noise.
        actual_real_sales = min(self.inventory, self.expected_demand * np.random.uniform(0.9, 1.1))
        self.inventory -= actual_real_sales
        
        nominal_revenue = actual_real_sales * price_level * trade_multiplier
        self.output = nominal_revenue  
        
        # 3. Supply Chain Intermediate Consumption
        intermediate_cost = 0.0
        for input_sec, req_pct in self.io_reqs.items():
            base_cost = nominal_revenue * req_pct
            
            # Use Supply Chain Network for Paton Admissibility routing
            if hasattr(self.model, 'supply_chain'):
                routed_cost = self.model.supply_chain.evaluate_supply_shock(self, input_sec, base_cost)
                intermediate_cost += routed_cost
            else:
                supply_multiplier = self.model.supply_multipliers.get(input_sec, 1.0)
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
        # Depreciation is unconditionally applied
        self.capital -= (depreciation_cost / price_level)
        
        # Minsky-Keen Investment Dynamics
        # Calculate Profit Rate (pi_r)
        if nominal_revenue > 0:
            wage_share = wage_bill / nominal_revenue
            debt_ratio = self.debt / nominal_revenue
        else:
            wage_share = 1.0
            debt_ratio = 1.0
            
        profit_rate = 1.0 - wage_share - (self.interest_rate * debt_ratio)
        
        # Logistic Investment Function kappa(pi_r)
        # Investment increases as profit rate rises, bounded by upper limit
        kappa = self.investment_rate / (1.0 + np.exp(-5.0 * (profit_rate - 0.05)))
        
        real_investment_desired = self.capital * kappa
        nominal_investment_desired = real_investment_desired * price_level
        
        # Financing the Investment (SFC constraints)
        retained_earnings = max(0, self.profit)
        external_financing_needed = nominal_investment_desired - retained_earnings
        
        if external_financing_needed > 0:
            # Request loan from the banking sector
            loan_approved = self.model.commercial_bank.request_loan(self, external_financing_needed)
            if loan_approved:
                actual_nominal_investment = nominal_investment_desired
            else:
                actual_nominal_investment = retained_earnings # Credit rationed
        else:
            actual_nominal_investment = nominal_investment_desired
            # Pay down debt with remaining profit
            debt_paydown = min(self.debt, retained_earnings - nominal_investment_desired)
            self.model.commercial_bank.process_repayment(self, debt_paydown)
            
        # Execute Investment
        self.capital += (actual_nominal_investment / price_level)
        
        # Deduct used deposits (retained earnings portion)
        self.deposits -= actual_nominal_investment
        if self.deposits < 0: self.deposits = 0
        
        # Labor firing if capacity heavily exceeds demand
        if target_production < max_capacity * 0.8:
            # Fire 10% of workers
            fire_count = int(self.labor * 0.1)
            self.labor = max(1, self.labor - fire_count)
            # Find and update households
            for hh in self.model.households:
                if hh.employer == self and fire_count > 0:
                    hh.employed = False
                    hh.employer = None
                    hh.wage = 0.0
                    fire_count -= 1
            self.debt += abs(self.profit)
            
            # Bankruptcy Check (Minsky Ponzi collapse)
            if self.debt > (self.capital * price_level * 2.0) or self.deposits < 0:
                self.bankruptcies += 1
                # Bank writes off the loan
                self.model.commercial_bank.loans -= self.debt
                self.model.commercial_bank.net_worth -= self.debt
                self.model.agents.remove(self)
                return  # Exit step
                
        self.capital = max(0.01, self.capital)
        self.labor = max(1.0, self.labor)
        self.tfp *= (1.0 + self.tfp_growth_rate)


def compute_state_output(model):
    """Aggregate nominal output by state."""
    state_totals = {}
    for agent in model.agents:
        if hasattr(agent, 'state') and hasattr(agent, 'output'):
            state_totals[agent.state] = state_totals.get(agent.state, 0.0) + agent.output
    return state_totals

class IndianEconomyModel(Model):
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
        
        # Macro variables
        self.price_level = 1.0
        self.inflation_rate = config['monetary_policy']['target_inflation_rate']
        self.repo_rate = config['monetary_policy']['rbi_repo_rate']
        self.total_tax_collection = 0.0
        
        self.capex_budgeted_crore = config.get('government', {}).get('capex_budgeted_crore', 0.0)
        
        # IO Supply Tracking
        self.supply_multipliers = {
            "Agriculture": 1.0,
            "Manufacturing": 1.0,
            "Services": 1.0
        }
        self.previous_supply = None
        self.baseline_supply = None
        
        from src.engine.sectors import CommercialBankAgent, CentralBankAgent, HouseholdAgent
        from src.engine.labor_market import LaborMarket
        from src.engine.space import UrbanSpace
        from src.engine.land_market import LandMarket
        from src.engine.demographics import MigrationEngine
        from src.engine.kinetics import KineticWealthExchange
        from src.engine.financial_markets import LimitOrderBook
        from src.engine.supply_chain import SupplyChainNetwork
        
        # Initialize Spatial Grid and Land Market
        self.grid = UrbanSpace(100, 100, torus=False)
        self.grid.initialize_ntl_proxy()
        self.land_market = LandMarket(self)
        self.migration_engine = MigrationEngine(self)
        self.kinetics = KineticWealthExchange(self)
        self.stock_market = LimitOrderBook(self)
        self.supply_chain = SupplyChainNetwork(self)
        
        # Initialize SFC Institutional Sectors
        self.central_bank = CentralBankAgent(self, "RBI_01")
        self.commercial_bank = CommercialBankAgent(self, "BANK_01")
        
        self.labor_market = LaborMarket(self)
        
        self.aggregate_consumption = 0.0
        
        # Create agents
        for idx, row in synthetic_firms_df.iterrows():
            debt = row.get("Debt", 0.0)
            firm = FirmAgent(
                model=self,
                unique_id=row['CIN'],
                sector=row['Sector'],
                state=row['State'],
                capital=row['Capital'],
                labor=row['Labor'],
                cap_share=row['Cap_share'],
                lab_share=row['Lab_share'],
                debt=debt,
                shocks=self.policy_shocks
            )
            
            # Place firm on Commercial grid cells if available, else random
            placed = False
            for _ in range(10):
                x, y = np.random.randint(100), np.random.randint(100)
                if self.grid.zoning[x][y] == 2: # Commercial
                    self.grid.place_agent(firm, (x, y))
                    placed = True
                    break
            if not placed:
                self.grid.place_agent(firm, (np.random.randint(100), np.random.randint(100)))
            
        # Create households (simplified Zipf distribution proxy)
        num_households = self.num_agents * 10
        self.households = []
        for i in range(num_households):
            hh = HouseholdAgent(self, f"HH_{i}", initial_wealth=100.0, reservation_wage=config['agent_logic']['baseline_wage_rate'], skill_level=1.0)
            self.households.append(hh)
            self.commercial_bank.deposits += 100.0
            
            # Place household on Residential grid cells
            placed = False
            for _ in range(10):
                x, y = np.random.randint(100), np.random.randint(100)
                if self.grid.zoning[x][y] == 1: # Residential
                    self.grid.place_agent(hh, (x, y))
                    placed = True
                    break
            if not placed:
                self.grid.place_agent(hh, (np.random.randint(100), np.random.randint(100)))
            
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Output": lambda m: sum([getattr(a, 'output', 0.0) for a in m.agents]),
                "Total_Profit": lambda m: sum([getattr(a, 'profit', 0.0) for a in m.agents]),
                "Total_Capital": lambda m: sum([getattr(a, 'capital', 0.0) for a in m.agents]),
                "Total_Labor": lambda m: sum([getattr(a, 'labor', 0) for a in m.agents]),
                "Total_Debt": lambda m: sum([getattr(a, 'debt', 0.0) for a in m.agents]),
                "Total_Tax_Revenue": lambda m: m.total_tax_collection,
                "Price_Level": lambda m: m.price_level,
                "Bankruptcies": lambda m: sum([getattr(a, 'bankruptcies', 0) for a in m.agents]),
                "Gini_Coefficient": compute_gini,
                "Agri_Output": lambda m: sum([getattr(a, 'output', 0.0) for a in m.agents if getattr(a, 'sector', None) == "Agriculture"]),
                "Mfg_Output": lambda m: sum([getattr(a, 'output', 0.0) for a in m.agents if getattr(a, 'sector', None) == "Manufacturing"]),
                "Svc_Output": lambda m: sum([getattr(a, 'output', 0.0) for a in m.agents if getattr(a, 'sector', None) == "Services"]),
                "Agri_Profit": lambda m: sum([getattr(a, 'profit', 0.0) for a in m.agents if getattr(a, 'sector', None) == "Agriculture"]),
                "Mfg_Profit": lambda m: sum([getattr(a, 'profit', 0.0) for a in m.agents if getattr(a, 'sector', None) == "Manufacturing"]),
                "Svc_Profit": lambda m: sum([getattr(a, 'profit', 0.0) for a in m.agents if getattr(a, 'sector', None) == "Services"]),
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
        
        # Calculate Macro Bank Step
        self.commercial_bank.step()
        self.central_bank.step()
        self.aggregate_consumption = 0.0
        
        # Step households (Kinetic trade / consumption)
        for hh in self.households:
            hh.step()
            
        # Step all firm agents through the Mesa scheduler
        self.agents.shuffle_do("step")
        
        # Run Labor Market matching
        self.labor_market.step()
        
        # Run Land Market auctions
        self.land_market.step()
        
        # Run Demographic Migration
        self.migration_engine.step()
        
        # Run Kinetic Wealth Exchange (Trade/Consumption and DBTs)
        self.kinetics.step()
        
        # Run Stock Market Double Auction
        self.stock_market.step()
        
        # Reset Supply Chain bottleneck trackers
        self.supply_chain.step()
            
        # Calculate sector supply for the next tick's intermediate consumption
        current_supply = {
            "Agriculture": sum([getattr(a, 'output', 0.0) for a in self.agents if getattr(a, 'sector', None) == "Agriculture"]),
            "Manufacturing": sum([getattr(a, 'output', 0.0) for a in self.agents if getattr(a, 'sector', None) == "Manufacturing"]),
            "Services": sum([getattr(a, 'output', 0.0) for a in self.agents if getattr(a, 'sector', None) == "Services"])
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
        
        # Distribute Capex back into the economy as demand (increases revenue of random firms)
        firms_only = [a for a in self.agents if getattr(a, 'sector', None) is not None]
        if self.capex_budgeted_crore > 0 and len(firms_only) > 0:
            capex_per_firm = self.capex_budgeted_crore / len(firms_only)
            for agent in firms_only:
                agent.output += capex_per_firm
