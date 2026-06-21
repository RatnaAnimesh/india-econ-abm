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
        Enforces SFC: Buyer transfers deposits to Seller (and transport cost to Services).
        """
        from src.engine.model import FirmAgent
        
        # Find firms in the input_sec using strict isinstance filtering
        suppliers = [f for f in self.model.agents if isinstance(f, FirmAgent) and f.sector == input_sec and f.inventory > 0]
        
        if not suppliers:
            # Complete system failure for this input
            self.capacity_bottlenecks += 1
            exec_cost = base_cost * 5.0 # Max penalty
            
            # Deduct from buyer, money goes to central pool or bank (no specific supplier, so bank takes it as fee)
            firm.deposits -= exec_cost
            if firm.deposits < 0:
                firm.debt += abs(firm.deposits)
                firm.deposits = 0.0
            return exec_cost
            
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
            real_volume = min(best_supplier.inventory, base_cost / max(1.0, self.model.price_level))
            best_supplier.inventory = max(0.0, best_supplier.inventory - real_volume)
            
            # SFC Transfer: Buyer pays Supplier and Transport provider
            firm.deposits -= exec_cost
            if firm.deposits < 0:
                firm.debt += abs(firm.deposits)
                firm.deposits = 0.0
                
            best_supplier.deposits += base_cost
            best_supplier.intermediate_revenue += base_cost
            best_supplier.output += base_cost
            
            # Route transport cost (markup) to Services sector
            transport_cost = exec_cost - base_cost
            if transport_cost > 0:
                services_firms = [f for f in self.model.agents if isinstance(f, FirmAgent) and f.sector == "Services"]
                if services_firms:
                    transport_firm = np.random.choice(services_firms)
                    transport_firm.deposits += transport_cost
                    transport_firm.intermediate_revenue += transport_cost
                    transport_firm.output += transport_cost
            
            return exec_cost
        else:
            self.capacity_bottlenecks += 1
            exec_cost = base_cost * 3.0 # Rerouting penalty
            
            firm.deposits -= exec_cost
            if firm.deposits < 0:
                firm.debt += abs(firm.deposits)
                firm.deposits = 0.0
            return exec_cost
            
    def step(self):
        # Reset bottleneck tracker
        self.capacity_bottlenecks = 0
