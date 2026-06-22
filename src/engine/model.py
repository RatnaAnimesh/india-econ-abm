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
    """Compute the Gini coefficient of firm capital distribution."""
    # Use pre-filtered active_firms for speed
    active_firms = getattr(model, 'active_firms', [])
    if not active_firms:
        return 0
    agent_wealths = [a.capital for a in active_firms]
    x = sorted(agent_wealths)
    N = len(agent_wealths)
    if N == 0 or sum(x) == 0:
        return 0
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N*sum(x))
    return (1 + (1/N) - 2*B)

class FirmAgent(Agent):
    """A firm in the Indian economy."""
    def __init__(self, model, unique_id, sector, state, capital, cap_share, lab_share, debt, shocks=None, labor=1.0):
        super().__init__(model)
        self.cin = unique_id  # Store the CIN since Mesa 3 generates its own unique_id
        self.sector = sector
        self.state = state
        self.capital = capital
        self.cap_share = cap_share
        self.lab_share = lab_share
        self.debt = debt
        self.deposits = debt * 0.1 # initial liquidity buffer
        self.employees = []
        self.intermediate_revenue = 0.0
        self.final_revenue = 0.0
        self.shocks = shocks or {}
        
        # Load empirical parameters from config
        self.wage_rate = config['agent_logic']['baseline_wage_rate'] * np.random.uniform(0.9, 1.1)
        self.depreciation_rate = config['agent_logic']['depreciation_rate']
        self.investment_rate = config['agent_logic']['investment_rate']
        self.profit_labor_growth_rate = config['agent_logic']['profit_labor_growth_rate']
        self.loss_labor_shrink_rate = config['agent_logic']['loss_labor_shrink_rate']
        # TFP growth rate with firm-level variation
        baseline_tfp_growth = config['agent_logic']['tfp_growth_rates'][self.sector] / 100.0
        self.tfp_growth_rate = max(0.0, baseline_tfp_growth + np.random.normal(0.0, 0.002))
        
        # Fiscal Policy
        self.corporate_tax_rate = config['fiscal_policy']['corporate_tax_rate']
        try:
            division = int(self.cin[1:3])
        except (ValueError, TypeError, IndexError):
            division = 0
            
        gst_tiers = config['fiscal_policy']['gst_rates']
        if division in [1, 2, 3, 84, 85, 86, 87, 88]:
            self.gst_rate = gst_tiers.get('exempt', 0.00)
        elif division in [10, 11, 13, 14, 15, 55, 56]:
            self.gst_rate = gst_tiers.get('basic', 0.05)
        elif division in [12, 29, 30]:
            self.gst_rate = gst_tiers.get('demerit', 0.40)
        else:
            self.gst_rate = gst_tiers.get('standard', 0.18)
        
        # Fetch interest rate based on global repo rate
        self.interest_rate = max(0.0, self.model.repo_rate + config['monetary_policy']['corporate_spread'])
        
        # Trade Policy
        self.export_propensity = config['trade_policy']['export_propensity'][self.sector]
        
        # IO Matrix Requirements
        self.io_reqs = {
            "Agriculture": config['io_matrix']['Agriculture'][self.sector],
            "Manufacturing": config['io_matrix']['Manufacturing'][self.sector],
            "Services": config['io_matrix']['Services'][self.sector]
        }
        
        # State variables for tracking
        self.output = 0.0
        self.profit = 0.0
        self.tfp = 1.0 # Total Factor Productivity (A)
        self.bankruptcies = 0
        self.intermediate_cost = 0.0
        self.wages_paid = 0.0
        self.interest_payment = 0.0
        self.gst_payment = 0.0
        self.depreciation_cost = 0.0
        self.corporate_tax = 0.0
        self.ebt = 0.0
        
        # Adaptive Expectations & Production
        self.expected_demand = 0.0
        self.inventory = 0.0
        self.learning_rate = 0.2 # eta
        self.buffer_ratio = 0.1 # nu
        self.bankrupt = False
        self.carbon_tax_payment = 0.0
        
    @property
    def labor(self):
        return max(1.0, float(len(self.employees)))
        
    @property
    def net_worth(self):
        return self.capital + self.deposits - self.debt
        
    @property
    def emission_intensity(self):
        if not self.pos:
            return 0.0
        sec_intensity = {"Agriculture": 0.8, "Manufacturing": 2.5, "Services": 0.3}
        grid_int = self.model.grid.grid_emission_intensity[self.pos[0]][self.pos[1]]
        return grid_int * sec_intensity.get(self.sector, 1.0)
        
    @property
    def cbam_exposure(self):
        if not self.pos:
            return 0.0
        return self.model.grid.cbam_exposure[self.pos[0]][self.pos[1]]
        
    @property
    def climate_vulnerability(self):
        if not self.pos:
            return 0.0
        return self.model.grid.climate_vulnerability_score[self.pos[0]][self.pos[1]]
        
    @property
    def state_gsdp_per_capita(self):
        if not self.pos:
            return 0.0
        return self.model.grid.state_gsdp_per_capita[self.pos[0]][self.pos[1]]
        
    def produce(self):
        """Phase 1: Determine target demand and produce goods."""
        if self.expected_demand == 0.0:
            self.expected_demand = self.capital * 0.5 # initial heuristic
            
        sales_last_period = self.output / max(1.0, self.model.price_level) # Real sales
        self.expected_demand = self.expected_demand + self.learning_rate * (sales_last_period - self.expected_demand)
        
        target_production = max(0.0, self.expected_demand * (1.0 + self.buffer_ratio) - self.inventory)
        max_capacity = self.tfp * (self.capital ** self.cap_share) * (self.labor ** self.lab_share)
        real_output = min(target_production, max_capacity)
        
        self.inventory += real_output
        
    def export_sales(self):
        """Phase 2: Export goods to rest of the world (exogenous cash injection)."""
        price_level = self.model.price_level
        base_exchange_shock = config['trade_policy']['exchange_rate_shock']
        exchange_shock = base_exchange_shock + getattr(self.model, 'exchange_rate_shock_val', 0.0)
        
        # Export demand based on propensity and exchange rate
        export_demand = self.expected_demand * self.export_propensity * exchange_shock
        
        # Apply CBAM shock if active
        cbam_tariff = getattr(self.model, 'cbam_tariff', 0.0)
        if cbam_tariff > 0.0:
            cbam_cost_factor = cbam_tariff * self.emission_intensity * self.cbam_exposure
            cbam_cost_factor = np.clip(cbam_cost_factor, 0.0, 0.8)
            export_demand *= (1.0 - cbam_cost_factor)
            
        real_exports = min(self.inventory, export_demand)
        self.inventory = max(0.0, self.inventory - real_exports)
        
        export_rev = real_exports * price_level * exchange_shock
        if cbam_tariff > 0.0:
            # EU collects the tariff, resulting in leakage from export revenue
            cbam_cost_factor = cbam_tariff * self.emission_intensity * self.cbam_exposure
            cbam_cost_factor = np.clip(cbam_cost_factor, 0.0, 0.8)
            export_rev *= (1.0 - cbam_cost_factor)
            
        self.deposits += export_rev
        self.model.commercial_bank._deposits += export_rev
        self.model.commercial_bank.reserves += export_rev # NEW: Exports bring in Central Bank reserves
        self.final_revenue += export_rev
        self.output += export_rev
        
    def purchase_inputs(self):
        """Phase 3: Route supply chain intermediate demands to other firms."""
        self.intermediate_cost = 0.0
        for input_sec, req_pct in self.io_reqs.items():
            base_cost = self.output * req_pct
            if base_cost <= 0.0:
                continue
                
            if hasattr(self.model, 'supply_chain'):
                routed_cost = self.model.supply_chain.evaluate_supply_shock(self, input_sec, base_cost)
                self.intermediate_cost += routed_cost
            else:
                supply_multiplier = self.model.supply_multipliers.get(input_sec, 1.0)
                scarcity_price_modifier = min(5.0, 1.0 / max(0.2, supply_multiplier))
                self.intermediate_cost += (base_cost * scarcity_price_modifier)
                
    def pay_financials(self):
        """Phase 4: Pay wages, interest, and taxes."""
        price_level = self.model.price_level
        
        # 0. Pay Intermediate Supply Chain Costs
        self.deposits -= self.intermediate_cost
        # If we assume closed loop, this money would technically credit the supplier's deposits here.
        # For now, we must at least drain it from the buyer to prevent phantom liquidity.
        if self.deposits < 0.0:
            overdraft = abs(self.deposits)
            self.debt += overdraft
            self.model.commercial_bank._loans += overdraft
            self.model.commercial_bank._deposits += overdraft
            self.deposits = 0.0

        # 1. Pay Wages (directly to employees' deposits)
        self.wages_paid = 0.0
        for emp in self.employees:
            wage_to_pay = emp.wage
            self.deposits -= wage_to_pay
            emp.deposits += wage_to_pay
            self.wages_paid += wage_to_pay
            
        # Bank overdraft coverage
        if self.deposits < 0.0:
            overdraft = abs(self.deposits)
            self.debt += overdraft
            self.model.commercial_bank._loans += overdraft
            self.model.commercial_bank._deposits += overdraft
            self.deposits = 0.0
            
        # 2. Pay Interest
        self.interest_payment = self.debt * self.interest_rate
        self.deposits -= self.interest_payment
        # O(1) state bank update for interest payment (bank deposits decrease, net worth increases)
        self.model.commercial_bank._deposits -= self.interest_payment
        if self.deposits < 0.0:
            overdraft = abs(self.deposits)
            self.debt += overdraft
            self.model.commercial_bank._loans += overdraft
            self.model.commercial_bank._deposits += overdraft
            self.deposits = 0.0
            
        # 3. GST Payment
        self.gst_payment = self.output * self.gst_rate
        self.deposits -= self.gst_payment
        self.model.commercial_bank._deposits -= self.gst_payment
        self.model.commercial_bank.reserves -= self.gst_payment
        if self.deposits < 0.0:
            overdraft = abs(self.deposits)
            self.debt += overdraft
            self.model.commercial_bank._loans += overdraft
            self.model.commercial_bank._deposits += overdraft
            self.deposits = 0.0
        self.model.total_tax_collection += self.gst_payment
        
        # 3b. Carbon Tax Payment
        carbon_price = getattr(self.model, 'carbon_price', 0.0)
        self.carbon_tax_payment = 0.0
        if carbon_price > 0.0:
            emissions = self.emission_intensity * self.output
            self.carbon_tax_payment = emissions * carbon_price
            self.deposits -= self.carbon_tax_payment
            self.model.commercial_bank._deposits -= self.carbon_tax_payment
            self.model.commercial_bank.reserves -= self.carbon_tax_payment
            if self.deposits < 0.0:
                overdraft = abs(self.deposits)
                self.debt += overdraft
                self.model.commercial_bank._loans += overdraft
                self.model.commercial_bank._deposits += overdraft
                self.deposits = 0.0
            self.model.total_tax_collection += self.carbon_tax_payment
            
        # 4. Depreciation (non-cash charge, reduces capital value)
        self.depreciation_cost = self.capital * self.depreciation_rate * price_level
        self.capital = max(0.01, self.capital - (self.depreciation_cost / price_level))
        
        # 5. Earnings Before Tax (EBT)
        self.ebt = self.output - self.intermediate_cost - self.wages_paid - self.depreciation_cost - self.interest_payment - self.gst_payment - self.carbon_tax_payment
        
        # 6. Corporate Tax & Net Profit
        if self.ebt > 0:
            self.corporate_tax = self.ebt * self.corporate_tax_rate
            self.deposits -= self.corporate_tax
            self.model.commercial_bank._deposits -= self.corporate_tax
            self.model.commercial_bank.reserves -= self.corporate_tax
            if self.deposits < 0.0:
                overdraft = abs(self.deposits)
                self.debt += overdraft
                self.model.commercial_bank._loans += overdraft
                self.model.commercial_bank._deposits += overdraft
                self.deposits = 0.0
            self.model.total_tax_collection += self.corporate_tax
            self.profit = self.ebt - self.corporate_tax
        else:
            self.corporate_tax = 0.0
            self.profit = self.ebt
            
    def invest_and_finance(self):
        """Phase 5: Minsky investment demand, loan financing, and demographic updates."""
        price_level = self.model.price_level
        
        # Calculate Profit Rate (pi_r)
        if self.output > 0:
            wage_share = self.wages_paid / self.output
            debt_ratio = self.debt / self.output
        else:
            wage_share = 1.0
            debt_ratio = 1.0
            
        profit_rate = 1.0 - wage_share - (self.interest_rate * debt_ratio)
        
        # Logistic Investment Function kappa(pi_r)
        kappa = self.investment_rate / (1.0 + np.exp(-5.0 * (profit_rate - 0.05)))
        real_investment_desired = self.capital * kappa
        nominal_investment_desired = real_investment_desired * price_level
        
        retained_earnings = max(0.0, self.profit)
        external_financing_needed = nominal_investment_desired - retained_earnings
        
        if external_financing_needed > 0:
            # Request loan from the banking sector
            loan_approved = self.model.commercial_bank.request_loan(self, external_financing_needed)
            if loan_approved:
                actual_nominal_investment = nominal_investment_desired
            else:
                actual_nominal_investment = retained_earnings # rationed
        else:
            actual_nominal_investment = nominal_investment_desired
            # Pay down debt with remaining profit
            debt_paydown = min(self.debt, retained_earnings - nominal_investment_desired)
            self.model.commercial_bank.process_repayment(self, debt_paydown)
            
        # Execute Investment
        self.capital += (actual_nominal_investment / price_level)
        self.deposits -= actual_nominal_investment
        
        # Distribute the investment capital to a random supplier to close the SFC loop
        if self.model.active_firms:
            import random
            supplier = random.choice(self.model.active_firms)
            supplier.deposits += actual_nominal_investment
            if not hasattr(supplier, 'final_revenue'):
                supplier.final_revenue = 0.0
            supplier.final_revenue += actual_nominal_investment
        # The money transfers to another firm's deposit account within the same bank.
        # Therefore, aggregate bank deposits and reserves do NOT decrease.
        if self.deposits < 0.0:
            overdraft = abs(self.deposits)
            self.debt += overdraft
            self.model.commercial_bank._loans += overdraft
            self.model.commercial_bank._deposits += overdraft
            self.deposits = 0.0
            
        # Fire labor if target production falls below capacity
        max_capacity = self.tfp * (self.capital ** self.cap_share) * (self.labor ** self.lab_share)
        target_prod = self.expected_demand * (1.0 + self.buffer_ratio) - self.inventory
        if target_prod < max_capacity * 0.8 and len(self.employees) > 1:
            fire_count = max(1, int(len(self.employees) * 0.1))
            for _ in range(fire_count):
                if self.employees:
                    hh = self.employees.pop()
                    hh.employed = False
                    hh.employer = None
                    hh.wage = 0.0
                    
        # Bankruptcy check (debt outgrows collateral by 2x)
        if self.debt > (self.capital * price_level * 2.0):
            self.bankruptcies += 1
            # O(1) state bank update for loan write-off
            self.model.commercial_bank._loans -= self.debt
            self.model.commercial_bank._deposits -= self.deposits
            
            # Set bankrupt firm's financials to zero
            self.debt = 0.0
            self.deposits = 0.0
            self.bankrupt = True
            
            # Release employees
            for hh in self.employees:
                hh.employed = False
                hh.employer = None
                hh.wage = 0.0
            self.employees = []
            self.model.agents.remove(self)
            return
            
        self.capital = max(0.01, self.capital)
        self.tfp *= (1.0 + self.tfp_growth_rate)

def compute_state_output(model):
    """Aggregate nominal output by state."""
    state_totals = {}
    active_firms = getattr(model, 'active_firms', [])
    for agent in active_firms:
        state_totals[agent.state] = state_totals.get(agent.state, 0.0) + agent.output
    return state_totals

def compute_state_emissions(model):
    """Aggregate emissions by state."""
    state_totals = {}
    active_firms = getattr(model, 'active_firms', [])
    for agent in active_firms:
        state_totals[agent.state] = state_totals.get(agent.state, 0.0) + (agent.emission_intensity * agent.output)
    return state_totals

def compute_state_carbon_tax(model):
    """Aggregate carbon tax by state."""
    state_totals = {}
    active_firms = getattr(model, 'active_firms', [])
    for agent in active_firms:
        state_totals[agent.state] = state_totals.get(agent.state, 0.0) + getattr(agent, 'carbon_tax_payment', 0.0)
    return state_totals

class IndianEconomyModel(Model):
    """The macro-economy model containing all firms and government."""
    def __init__(self, data_path=None, policy_shocks=None, seed=None):
        if seed is None:
            seed = config['run'].get('master_seed', 42)
            
        super().__init__(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.policy_shocks = policy_shocks or []
        self.current_tick = 0
        
        import pandas as pd
        if data_path:
            synthetic_firms_df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide data_path to synthetic firms.")
            
        self.num_agents = len(synthetic_firms_df)
        
        # Macro variables
        self.price_level = 1.0
        self.inflation_rate = config['monetary_policy']['target_inflation_rate']
        self.total_tax_collection = 0.0
        self.exchange_rate_shock_val = 0.0
        self.carbon_price = 0.0
        self.cbam_tariff = 0.0
        
        # Synchronized repo rate via macro data loader
        from src.data_pipeline.data_loader import DataLoader
        loader = DataLoader(config_path=config_path)
        macro_data = loader.fetch_macro_data(allow_stale=True)
        
        self.repo_rate = config['monetary_policy']['rbi_repo_rate']
        if "repo_rate" in macro_data and "value" in macro_data["repo_rate"]:
            self.repo_rate = macro_data["repo_rate"]["value"]
            
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
        
        # Initialize Spatial Grid and Land Market with dynamic bounds
        cell_size = config['grid'].get('cell_size_deg', 0.25)
        lat_min, lat_max = config['grid']['lat_bounds']
        lon_min, lon_max = config['grid']['lon_bounds']
        width = int((lon_max - lon_min) / cell_size)
        height = int((lat_max - lat_min) / cell_size)
        
        self.grid = UrbanSpace(width, height, torus=False)
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
        
        # Create agents (firms)
        self.firms = []
        for idx, row in synthetic_firms_df.iterrows():
            # Cap debt at 1.5x capital to prevent immediate day-zero bankruptcy
            # The bankruptcy threshold is 2.0x capital.
            raw_debt = row.get("Debt", 0.0)
            max_allowed_debt = row['Capital'] * 1.5
            debt = min(raw_debt, max_allowed_debt)
            firm = FirmAgent(
                model=self,
                unique_id=row['CIN'],
                sector=row['Sector'],
                state=row['State'],
                capital=row['Capital'],
                cap_share=row['Cap_share'],
                lab_share=row['Lab_share'],
                debt=debt,
                shocks=None,
                labor=row['Labor']
            )
            self.firms.append(firm)
            
            # Place firm in its registered state
            comm_cells = self.grid.cells_by_state_and_zoning.get((firm.state, 2), [])
            if comm_cells:
                pos = self.random.choice(comm_cells)
            else:
                state_cells = self.grid.cells_by_state.get(firm.state, [])
                if state_cells:
                    pos = self.random.choice(state_cells)
                else:
                    mp_comm = self.grid.cells_by_state_and_zoning.get(("Madhya Pradesh", 2), [])
                    if mp_comm:
                        pos = self.random.choice(mp_comm)
                    else:
                        pos = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))
            self.grid.place_agent(firm, pos)
            
        # Create households and match to initial firm labor requirements
        firm_target_labor = {}
        for idx, row in synthetic_firms_df.iterrows():
            req_labor = max(1, int(np.round(row['Labor'])))
            firm_target_labor[row['CIN']] = req_labor
            
        # Calibrate initial unemployment to the PLFS target of ~4.9% (total labor supply is ~1.052x target labor)
        sum_target_labor = sum(firm_target_labor.values())
        num_households = int(sum_target_labor * 1.052)
        self.households = []
            
        hh_idx = 0
        for firm in self.firms:
            target = firm_target_labor[firm.cin]
            for _ in range(target):
                if hh_idx < num_households:
                    hh = HouseholdAgent(self, f"HH_{hh_idx}", initial_wealth=100.0, reservation_wage=config['agent_logic']['baseline_wage_rate'], skill_level=1.0)
                    hh.employed = True
                    hh.employer = firm
                    hh.wage = firm.wage_rate * self.price_level
                    hh.state = firm.state
                    firm.employees.append(hh)
                    self.households.append(hh)
                    hh_idx += 1
                    
        # Remaining households initialized as unemployed
        while hh_idx < num_households:
            random_firm = self.random.choice(self.firms)
            hh = HouseholdAgent(self, f"HH_{hh_idx}", initial_wealth=100.0, reservation_wage=config['agent_logic']['baseline_wage_rate'], skill_level=1.0)
            hh.employed = False
            hh.employer = None
            hh.wage = 0.0
            hh.state = random_firm.state
            self.households.append(hh)
            hh_idx += 1
            
        # Place households on Residential cells in their state
        for hh in self.households:
            res_cells = self.grid.cells_by_state_and_zoning.get((hh.state, 1), [])
            if res_cells:
                pos = self.random.choice(res_cells)
            else:
                state_cells = self.grid.cells_by_state.get(hh.state, [])
                if state_cells:
                    pos = self.random.choice(state_cells)
                else:
                    mp_res = self.grid.cells_by_state_and_zoning.get(("Madhya Pradesh", 1), [])
                    if mp_res:
                        pos = self.random.choice(mp_res)
                    else:
                        pos = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))
            self.grid.place_agent(hh, pos)
            # Update state based on final location
            hh.state = self.grid.cell_states[hh.pos]
            
        # Initialize commercial bank deposits and loans state variables
        self.commercial_bank._loans = sum(firm.debt for firm in self.firms)
        self.commercial_bank._deposits = sum(firm.deposits for firm in self.firms) + sum(hh.deposits for hh in self.households)
        
        # Enforce commercial bank solvency and double-entry consistency on day zero.
        # Bank starts with a robust capital buffer (equity). Reserves are set to exactly match assets/liabilities.
        initial_bank_equity = 2000000.0
        self.commercial_bank.reserves = self.commercial_bank.deposits + initial_bank_equity - self.commercial_bank.loans
        self.central_bank.reserves = self.commercial_bank.reserves
        
        # Store active firms for day zero validation and model metrics
        self.active_firms = [a for a in self.agents if isinstance(a, FirmAgent)]
        
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Output": lambda m: sum([a.output for a in m.active_firms]),
                "Total_Profit": lambda m: sum([a.profit for a in m.active_firms]),
                "Total_Capital": lambda m: sum([a.capital for a in m.active_firms]),
                "Total_Labor": lambda m: sum([a.labor for a in m.active_firms]),
                "Total_Debt": lambda m: sum([a.debt for a in m.active_firms]),
                "Total_Tax_Revenue": lambda m: m.total_tax_collection,
                "Price_Level": lambda m: m.price_level,
                "Bankruptcies": lambda m: sum([a.bankruptcies for a in m.active_firms]),
                "Gini_Coefficient": compute_gini,
                "Agri_Output": lambda m: sum([a.output for a in m.active_firms if a.sector == "Agriculture"]),
                "Mfg_Output": lambda m: sum([a.output for a in m.active_firms if a.sector == "Manufacturing"]),
                "Svc_Output": lambda m: sum([a.output for a in m.active_firms if a.sector == "Services"]),
                "Agri_Profit": lambda m: sum([a.profit for a in m.active_firms if a.sector == "Agriculture"]),
                "Mfg_Profit": lambda m: sum([a.profit for a in m.active_firms if a.sector == "Manufacturing"]),
                "Svc_Profit": lambda m: sum([a.profit for a in m.active_firms if a.sector == "Services"]),
                "State_Output": compute_state_output,
                "Total_Emissions": lambda m: sum([a.emission_intensity * a.output for a in m.active_firms]),
                "Carbon_Tax_Revenue": lambda m: sum([getattr(a, 'carbon_tax_payment', 0.0) for a in m.active_firms]),
                "State_Emissions": compute_state_emissions,
                "State_Carbon_Tax": compute_state_carbon_tax,
                "Agri_Price_Multiplier": lambda m: 1.0 / max(0.2, m.supply_multipliers.get("Agriculture", 1.0)),
                "Mfg_Price_Multiplier": lambda m: 1.0 / max(0.2, m.supply_multipliers.get("Manufacturing", 1.0)),
                "Svc_Price_Multiplier": lambda m: 1.0 / max(0.2, m.supply_multipliers.get("Services", 1.0))
            }
        )

    def apply_shock(self, shock):
        """Applies a specific macroeconomic policy shock to the model or agents."""
        print(f"Applying policy shock: {shock}")
        shock_type = shock.get('type')
        val = shock.get('value', 0.0)
        
        if shock_type == 'repo_rate_shock':
            self.repo_rate = max(0.0, self.repo_rate + val)
            for firm in self.agents:
                if isinstance(firm, FirmAgent):
                    firm.interest_rate = max(0.0, self.repo_rate + config['monetary_policy']['corporate_spread'])
                    
        elif shock_type == 'gst_shock':
            sector = shock.get('sector', None)
            for firm in self.agents:
                if isinstance(firm, FirmAgent):
                    if sector is None or firm.sector == sector:
                        firm.gst_rate = max(0.0, firm.gst_rate + val)
                        
        elif shock_type == 'exchange_rate_shock':
            self.exchange_rate_shock_val += val
            
        elif shock_type == 'demonetisation_shock':
            for hh in self.households:
                if not hh.digital_adoption:
                    cash_loss = hh.deposits * 0.5 * val
                    hh.deposits -= cash_loss
                    self.commercial_bank._deposits -= cash_loss
                    self.commercial_bank.reserves -= cash_loss
                    
        elif shock_type == 'carbon_price_shock':
            self.carbon_price = max(0.0, self.carbon_price + val)
            
        elif shock_type == 'cbam_shock':
            self.cbam_tariff = max(0.0, self.cbam_tariff + val)

    def step(self):
        """Advance the model by one step (one year)."""
        self.price_level *= (1.0 + self.inflation_rate)
        self.total_tax_collection = 0.0
        self.aggregate_consumption = 0.0
        
        # Store active firms in self.active_firms for fast access in datacollector and step loops
        self.active_firms = [a for a in self.agents if isinstance(a, FirmAgent)]
        
        # Pre-group active firms by state for fast consumption O(1) lookups
        self.firms_by_state = {}
        for firm in self.active_firms:
            self.firms_by_state.setdefault(firm.state, []).append(firm)
            
        # 2. Reset step variables for all firms
        for firm in self.active_firms:
            firm.intermediate_revenue = 0.0
            firm.final_revenue = 0.0
            firm.output = 0.0
            firm.profit = 0.0
            
        # 1. Distribute Capex back into the economy as demand
        if self.capex_budgeted_crore > 0 and len(self.active_firms) > 0:
            capex_per_firm = self.capex_budgeted_crore / len(self.active_firms)
            for agent in self.active_firms:
                agent.deposits += capex_per_firm
                agent.output += capex_per_firm
                # Route capex to final_revenue as government consumption demand
                agent.final_revenue += capex_per_firm
                self.total_tax_collection -= capex_per_firm
                # Inform bank of deposit and reserves changes
                self.commercial_bank._deposits += capex_per_firm
                self.commercial_bank.reserves += capex_per_firm
            
        # Apply policy shocks scheduled for the current tick
        for shock in self.policy_shocks:
            if shock.get('tick', 0) == self.current_tick:
                self.apply_shock(shock)
                
        # 3. Production Phase
        for firm in self.active_firms:
            firm.produce()
            
        # 4. Consumption Phase
        for hh in self.households:
            hh.consume()
            
        # 5. Exports Phase
        for firm in self.active_firms:
            firm.export_sales()
            
        # 6. Intermediate Supply Chain Phase
        # Precompute suppliers by sector with inventory > 0 and services firms for fast lookups
        self.suppliers_by_sector = {
            "Agriculture": [],
            "Manufacturing": [],
            "Services": []
        }
        self.services_firms = []
        for a in self.active_firms:
            if a.inventory > 0:
                self.suppliers_by_sector[a.sector].append(a)
            if a.sector == "Services":
                self.services_firms.append(a)
                    
        for firm in self.active_firms:
            firm.purchase_inputs()
            
        # 7. Financials Phase (pay wages, interest, taxes, and calculate profit)
        for firm in self.active_firms:
            firm.pay_financials()
            
        for hh in self.households:
            hh.step()
            
        # 8. Investment Phase (Minsky capital expansion)
        for firm in self.active_firms:
            firm.invest_and_finance()
            
        # 9. Market / Labor / Space steps
        self.labor_market.step()
        self.land_market.step()
        self.migration_engine.step()
        self.kinetics.step() # wealth exchange between households
        self.stock_market.step()
        self.supply_chain.step()
        
        # Step bank and central bank
        self.commercial_bank.step()
        self.central_bank.reserves = self.commercial_bank.reserves
        self.central_bank.step()
        
        # Remove bankrupt firms from active_firms list before output calculations and data collection
        self.active_firms = [f for f in self.active_firms if not getattr(f, 'bankrupt', False)]
            
        # Calculate sector supply for the next tick's intermediate consumption
        current_supply = {
            "Agriculture": sum([a.output for a in self.active_firms if a.sector == "Agriculture"]),
            "Manufacturing": sum([a.output for a in self.active_firms if a.sector == "Manufacturing"]),
            "Services": sum([a.output for a in self.active_firms if a.sector == "Services"])
        }
        
        if self.baseline_supply is None:
            self.baseline_supply = current_supply
        else:
            for sec in current_supply:
                if self.baseline_supply[sec] > 0:
                    self.supply_multipliers[sec] = current_supply[sec] / self.baseline_supply[sec]
                    
        self.previous_supply = current_supply
        
        self.current_tick += 1
        self.datacollector.collect(self)
