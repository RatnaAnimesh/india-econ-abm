import numpy as np

class MigrationEngine:
    """
    Handles Rural-to-Urban migration via utility network functions.
    """
    def __init__(self, model):
        self.model = model
        
        # Utility parameters
        self.beta1 = 1.0 # Wage sensitivity
        self.beta2 = 0.5 # Price/cost sensitivity
        self.beta3 = 0.2 # Network sensitivity
        self.beta4 = 0.1 # Distance penalty
        
    def execute_migration(self):
        """
        Evaluate utility for agents in Cleared/Agri zones (0) 
        to move to Residential zones (1) near Commercial zones (2).
        """
        from src.engine.model import FirmAgent
        
        grid = self.model.grid
        if not hasattr(grid, 'zoning'):
            return
            
        rural_households = [hh for hh in self.model.households if hh.pos and grid.zoning[hh.pos[0]][hh.pos[1]] == 0]
        
        # Find all residential zones
        res_zones = []
        for x in range(grid.width):
            for y in range(grid.height):
                if grid.zoning[x][y] == 1:
                    res_zones.append((x, y))
                    
        if not res_zones:
            return
            
        # Precompute average wages by state for fast O(1) lookups
        avg_wage_by_state = {}
        for st, firms in getattr(self.model, 'firms_by_state', {}).items():
            active_firms_in_state = [f for f in firms if not f.bankrupt]
            if active_firms_in_state:
                avg_wage_by_state[st] = np.mean([f.wage_rate for f in active_firms_in_state]) * self.model.price_level
                
        for hh in rural_households:
            # Only consider migration if unemployed or poor
            if hh.employed and hh.wage > hh.reservation_wage * 1.5:
                continue
                
            # Sample potential destinations
            dest_sample = np.random.choice(len(res_zones), min(5, len(res_zones)), replace=False)
            
            best_dest = None
            best_utility = -np.inf
            
            for idx in dest_sample:
                dest = res_zones[idx]
                
                # Fetch average wage in destination cell's state or fallback
                dest_state = grid.cell_states.get(dest)
                expected_wage = avg_wage_by_state.get(dest_state, hh.reservation_wage * 1.2)
                
                # Proxy for cost of living (land price at dest)
                cost_of_living = grid.land_prices[dest[0]][dest[1]] * 0.01
                
                # Network effect: how many households already at dest?
                network_size = len(grid.get_cell_list_contents([dest]))
                
                # Distance
                dist = ((hh.pos[0] - dest[0])**2 + (hh.pos[1] - dest[1])**2)**0.5
                
                # Utility Function U_ij
                u = self.beta1 * expected_wage - self.beta2 * cost_of_living + self.beta3 * network_size - self.beta4 * dist
                
                # Add random Gumbel noise for logit choice
                u += np.random.gumbel()
                
                if u > best_utility:
                    best_utility = u
                    best_dest = dest
                    
            # Current utility (staying)
            current_network = len(grid.get_cell_list_contents([hh.pos]))
            u_stay = self.beta1 * hh.wage - self.beta2 * (grid.land_prices[hh.pos[0]][hh.pos[1]] * 0.01) + self.beta3 * current_network
            
            if best_utility > u_stay:
                # Migrate
                grid.move_agent(hh, best_dest)
                
    def step(self):
        self.execute_migration()
