from mesa import Agent
import random
import numpy as np

class HouseholdAgent(Agent):
    """
    Household agent for the Stock-Flow Consistent ABM.
    Provides labor, consumes goods, saves in bank deposits.
    """
    def __init__(self, model, unique_id, initial_wealth, reservation_wage, skill_level):
        super().__init__(model)
        # BSM Variables
        self.deposits = initial_wealth
        self.net_worth = initial_wealth
        self.loans = 0.0 # Typically 0 for households unless consumer credit is modeled
        
        # Labor Market Variables
        self.reservation_wage = reservation_wage
        self.skill_level = skill_level
        self.employer = None
        self.wage = 0.0
        self.employed = False
        
        # Consumption Variables
        self.mpc = 0.8 # Marginal Propensity to Consume
        
        # Spatial location metadata
        self.state = "Delhi"  # Will be mapped dynamically at model initialization
        
        # Demonetisation / Digital Adoption State
        self.trust = np.random.uniform(0.1, 0.9)
        self.security = np.random.uniform(0.1, 0.9)
        self.perceived_risk = np.random.uniform(0.1, 0.9)
        self.digital_adoption = False
        
    def evaluate_digital_adoption(self, is_shocked=False):
        """Sigmoid probability of digital adoption based on psychology and infrastructure."""
        if self.digital_adoption: return True
        
        # Infrastructure check
        grid = self.model.grid
        infra_locations = getattr(grid, 'digital_infrastructure', [])
        has_infra = False
        if self.pos and infra_locations:
            min_dist = min([((self.pos[0] - loc[0])**2 + (self.pos[1] - loc[1])**2)**0.5 for loc in infra_locations])
            has_infra = min_dist < 20.0 # Theta_distance
            
        if not has_infra:
            self.digital_adoption = False
            return False
            
        # Psychological Sigmoid
        beta1, beta2, beta3 = 1.5, 1.0, 2.0
        exponent = beta1 * self.trust + beta2 * self.security - beta3 * self.perceived_risk
        prob = 1.0 / (1.0 + np.exp(-exponent))
        
        if np.random.uniform(0, 1) < prob:
            self.digital_adoption = True
            
        # Post-shock learning
        if is_shocked:
            self.trust = min(1.0, self.trust * 1.05)
            self.security = min(1.0, self.security * 1.05)
            self.perceived_risk = max(0.0, self.perceived_risk * 0.95)
            
        return self.digital_adoption
        
    def consume(self):
        """Spend a portion of wealth on consumption of domestic firms in the same state."""
        if self.deposits <= 0.0:
            return
        
        c_exp = self.deposits * self.mpc
        if c_exp <= 0.0:
            return
            
        # Look up pre-grouped firms by state from the model
        state_firms = getattr(self.model, 'firms_by_state', {}).get(self.state, [])
        if not state_firms:
            # Fallback to all active firms
            state_firms = getattr(self.model, 'active_firms', [])
            
        if not state_firms:
            return # No firms to buy from
            
        # Select a firm to buy from (fully reproducible using model random)
        firm = self.model.random.choice(state_firms)
        
        # Determine actual purchase based on firm inventory
        price_level = self.model.price_level
        real_demand = c_exp / price_level
        
        real_purchase = min(firm.inventory, real_demand)
        if real_purchase > 0:
            nominal_purchase = real_purchase * price_level
            
            # SFC transfer
            self.deposits -= nominal_purchase
            firm.deposits += nominal_purchase
            
            firm.inventory = max(0.0, firm.inventory - real_purchase)
            firm.final_revenue += nominal_purchase
            firm.output += nominal_purchase
            
            self.model.aggregate_consumption += nominal_purchase

    def step(self):
        # Check for demonetisation shock in policy_shocks list
        is_shocked = False
        shock_level = 0.0
        for shock in getattr(self.model, 'policy_shocks', []):
            if shock.get('type') == 'demonetisation_shock' and shock.get('tick', 0) == self.model.current_tick:
                is_shocked = True
                shock_level = shock.get('value', 0.0)
                break
        
        self.evaluate_digital_adoption(is_shocked)
        
        # We do not apply cash loss here because it is applied directly in model.apply_shock()
        # to ensure exact single execution and reserves/deposits matching.
        
        # Receive interest on deposits
        interest_received = self.deposits * self.model.repo_rate * 0.95
        self.deposits += interest_received
        # Inform bank of deposit change
        self.model.commercial_bank._deposits += interest_received
            
        # Update Net Worth
        self.net_worth = self.deposits - self.loans


class CommercialBankAgent(Agent):
    """
    Commercial Bank agent.
    Creates loans ex-nihilo and manages deposits.
    Enforces SFC: Delta L = Delta D.
    """
    def __init__(self, model, unique_id):
        super().__init__(model)
        self.reserves = 0.0
        
        # State variables for deposits and loans (updated in O(1) during steps)
        self._loans = 0.0
        self._deposits = 0.0
        
        # Policy Constraints
        self.capital_adequacy_ratio = 0.08
        self.reserve_requirement = 0.03 # Cash Reserve Ratio (CRR) updated to 3.00%
        
    @property
    def loans(self):
        return self._loans

    @property
    def deposits(self):
        return self._deposits

    @property
    def net_worth(self):
        """Balance sheet identity: Equity = Assets - Liabilities."""
        return self.loans + self.reserves - self.deposits
        
    def request_loan(self, firm, amount):
        """
        Endogenous money creation:
        Bank credits firm's deposit account and its own loan book.
        """
        # Minskyan constraint: Banks ration credit if equity is too low relative to assets
        equity_ratio = self.net_worth / max(1.0, self.loans)
        
        # Simplified credit rationing: if equity ratio drops, bank denies loan
        if equity_ratio < self.capital_adequacy_ratio and self.loans > 0:
            return False
            
        # Create money ex-nihilo by crediting the firm's deposits and debt
        firm.deposits += amount
        firm.debt += amount
        
        # O(1) state update
        self._loans += amount
        self._deposits += amount
        return True
        
    def process_repayment(self, firm, amount):
        """Firm repays loan, destroying money."""
        actual_repayment = min(amount, firm.deposits, firm.debt)
        if actual_repayment > 0:
            firm.deposits -= actual_repayment
            firm.debt -= actual_repayment
            
            # O(1) state update
            self._loans -= actual_repayment
            self._deposits -= actual_repayment
        return actual_repayment

    def step(self):
        # CRR compliance check (minimum 3.00%)
        pass


class CentralBankAgent(Agent):
    """
    Reserve Bank of India (RBI).
    Sets repo rate and holds commercial bank reserves.
    """
    def __init__(self, model, unique_id):
        super().__init__(model)
        self.reserves = 0.0
        self.net_worth = 0.0
        
    def step(self):
        # Central Bank logic
        pass
