import numpy as np

class KineticWealthExchange:
    """
    Implements the Goswami-Sen kinetic exchange model for wealth distribution among households.
    """
    def __init__(self, model):
        self.model = model
        self.saving_propensity = 0.8 # lambda
        self.friction_threshold = 10.0 # Absolute poverty threshold
        
        # Policy
        self.wealth_tax_rate = 0.02
        self.dbt_pool = 0.0
        
    def execute_trades(self):
        """Random pairwise interactions simulating economic transactions."""
        num_households = len(self.model.households)
        if num_households < 2: return
        
        # Randomly pair up households
        # In a real spatial model, we'd weight by distance, but for simplicity we shuffle
        indices = np.random.permutation(num_households)
        
        for i in range(0, num_households - 1, 2):
            h1 = self.model.households[indices[i]]
            h2 = self.model.households[indices[i+1]]
            
            w1 = h1.deposits
            w2 = h2.deposits
            
            # Friction: If both are below poverty threshold, they don't trade
            if w1 < self.friction_threshold and w2 < self.friction_threshold:
                continue
                
            total_wealth = w1 + w2
            epsilon = np.random.uniform(0, 1)
            
            # Goswami-Sen equations
            new_w1 = self.saving_propensity * w1 + epsilon * (1 - self.saving_propensity) * total_wealth
            new_w2 = self.saving_propensity * w2 + (1 - epsilon) * (1 - self.saving_propensity) * total_wealth
            
            # Update deposits
            h1.deposits = new_w1
            h2.deposits = new_w2
            
            # Track consumption for macro aggregates
            # The traded amount is treated as consumption
            trade_vol = (1 - self.saving_propensity) * total_wealth
            self.model.aggregate_consumption += trade_vol

    def execute_fiscal_policy(self):
        """Wealth taxation and Direct Benefit Transfers (DBTs)."""
        # Sort households by wealth
        households = sorted(self.model.households, key=lambda x: x.deposits)
        num_households = len(households)
        if num_households == 0: return
        
        # Tax the top 10%
        top_10_idx = int(num_households * 0.9)
        top_households = households[top_10_idx:]
        
        tax_collected = 0.0
        for hh in top_households:
            tax = hh.deposits * self.wealth_tax_rate
            hh.deposits -= tax
            tax_collected += tax
            
        self.dbt_pool += tax_collected
        
        # Redistribute to the bottom 20%
        bottom_20_idx = int(num_households * 0.2)
        bottom_households = households[:bottom_20_idx]
        
        if len(bottom_households) > 0 and self.dbt_pool > 0:
            transfer_per_hh = self.dbt_pool / len(bottom_households)
            for hh in bottom_households:
                hh.deposits += transfer_per_hh
            self.dbt_pool = 0.0
            
    def step(self):
        self.execute_trades()
        self.execute_fiscal_policy()
