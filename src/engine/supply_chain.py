import numpy as np

class SupplyChainNetwork:
    """
    Evaluates spatial constraints and adaptive routing for intermediate consumption (IO network).
    """
    def __init__(self, model):
        self.model = model
        self.capacity_bottlenecks = 0
        
    def evaluate_supply_shock(self, firm, input_sec, base_cost):
        """
        Determines the effective cost of an input factoring in spatial capacity constraints.
        """
        # Find firms in the input_sec
        suppliers = [f for f in self.model.agents if hasattr(f, 'sector') and f.sector == input_sec and hasattr(f, 'inventory') and f.inventory > 0]
        
        if not suppliers:
            # Complete system failure for this input
            self.capacity_bottlenecks += 1
            return base_cost * 5.0 # Max penalty
            
        # Sample suppliers (spatial adaptive routing)
        sample_size = min(5, len(suppliers))
        sampled = np.random.choice(suppliers, sample_size, replace=False)
        
        best_supplier = None
        min_dist = np.inf
        
        for s in sampled:
            if firm.pos and s.pos:
                dist = ((firm.pos[0] - s.pos[0])**2 + (firm.pos[1] - s.pos[1])**2)**0.5
            else:
                dist = 50.0 # Default penalty
                
            # Admissibility constraint: Supplier must have enough inventory to cover a fraction of the cost
            # Assuming price_level approx 1 for simple volume check
            required_vol = base_cost / max(1.0, self.model.price_level)
            
            # If buffer falls below tau_buffer, reroute (dist penalty)
            tau_buffer = 10.0
            if s.inventory < tau_buffer:
                dist *= 2.0 # Congestion / low stock penalty
                
            if dist < min_dist and s.inventory > required_vol * 0.1:
                min_dist = dist
                best_supplier = s
                
        if best_supplier:
            # Execute physical routing
            exec_cost = base_cost * (1.0 + min_dist * 0.01) # Transport cost
            best_supplier.inventory = max(0, best_supplier.inventory - (base_cost / max(1.0, self.model.price_level)))
            return exec_cost
        else:
            self.capacity_bottlenecks += 1
            return base_cost * 3.0 # Rerouting penalty
            
    def step(self):
        # Reset bottleneck tracker
        self.capacity_bottlenecks = 0
