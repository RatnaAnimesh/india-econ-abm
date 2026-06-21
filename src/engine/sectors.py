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
        
    def step(self):
        # Check for demonetisation shock
        shock_level = getattr(self.model, 'policy_shocks', {}).get('demonetisation_shock', 0.0)
        is_shocked = shock_level > 0.0
        
        self.evaluate_digital_adoption(is_shocked)
        
        if is_shocked and not self.digital_adoption:
            # Unbanked cash is voided (86% of cash was demonetised, assume 50% loss for non-adopters)
            cash_loss = self.deposits * 0.5 * shock_level
            self.deposits -= cash_loss
            # This money leaves the system (shock constraint)
            
        # 1. Receive Wage (if employed)
        if self.employed and self.employer is not None:
            self.deposits += self.wage
            
        # 2. Update Net Worth
        self.net_worth = self.deposits - self.loans


class CommercialBankAgent(Agent):
    """
    Commercial Bank agent.
    Creates loans ex-nihilo and manages deposits.
    Enforces SFC: Delta L = Delta D.
    """
    def __init__(self, model, unique_id):
        super().__init__(model)
        self.loans = 0.0
        self.deposits = 0.0
        self.reserves = 0.0
        self.net_worth = 0.0
        
        # Policy Constraints
        self.capital_adequacy_ratio = 0.08
        self.reserve_requirement = 0.045 # Cash Reserve Ratio (CRR)
        
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
            
        # Create money ex-nihilo
        self.loans += amount
        self.deposits += amount
        
        # Transfer deposit to firm
        firm.deposits += amount
        firm.debt += amount
        
        return True
        
    def process_repayment(self, firm, amount):
        """Firm repays loan, destroying money."""
        actual_repayment = min(amount, firm.deposits, firm.debt)
        if actual_repayment > 0:
            firm.deposits -= actual_repayment
            firm.debt -= actual_repayment
            
            self.loans -= actual_repayment
            self.deposits -= actual_repayment
            
        return actual_repayment

    def step(self):
        # 1. Collect Interest on Loans
        interest_income = self.loans * self.model.repo_rate * 1.05 # Add spread
        
        # 2. Pay Interest on Deposits
        interest_expense = self.deposits * self.model.repo_rate * 0.95
        
        profit = interest_income - interest_expense
        self.net_worth += profit
        
        # Update Reserves with Central Bank
        required_reserves = self.deposits * self.reserve_requirement
        reserve_diff = required_reserves - self.reserves
        # Bank uses its net worth/deposits to meet reserve req
        self.reserves += reserve_diff
        self.model.central_bank.reserves += reserve_diff


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
        # Central Bank logic (e.g. Taylor Rule for repo rate adjustment)
        pass
