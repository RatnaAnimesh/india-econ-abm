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
            
        # Precompute local average wages for all residential cells to avoid slow nested spatial queries
        local_wages = {}
        for dest in res_zones:
            neighbors = grid.get_neighborhood(dest, moore=True, include_center=True, radius=10)
            local_firms = [a for a in grid.get_cell_list_contents(neighbors) if isinstance(a, FirmAgent)]
            if local_firms:
                local_wages[dest] = np.mean([f.wage_rate * self.model.price_level for f in local_firms])
            else:
                local_wages[dest] = None
                
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
                
                # Fetch precomputed local wage or fallback
                expected_wage = local_wages[dest]
                if expected_wage is None:
                    expected_wage = hh.reservation_wage * 1.2
                
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
