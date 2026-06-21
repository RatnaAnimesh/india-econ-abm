import numpy as np
from mesa import Agent

class RealEstateDeveloper(Agent):
    """
    Developer agent bidding for land using adaptive ratio alpha.
    """
    def __init__(self, model, unique_id):
        super().__init__(model)
        self.alpha = 1.0 # Bidding ratio
        self.delta = 0.05 # Learning rate
        self.prev_alpha = 1.0
        self.prev_profit = 0.0
        self.current_profit = 0.0
        self.capital = 10000.0
        
    def adapt_bidding_strategy(self):
        """Alpha adaptation based on profit gradients."""
        profit_diff = self.current_profit - self.prev_profit
        alpha_diff = self.alpha - self.prev_alpha
        
        sign_alpha = np.sign(alpha_diff) if alpha_diff != 0 else 1
        sign_profit = np.sign(profit_diff) if profit_diff != 0 else 1
        
        self.prev_alpha = self.alpha
        self.alpha = self.alpha + self.delta * sign_alpha * sign_profit
        self.alpha = np.clip(self.alpha, 0.5, 2.0)
        
        self.prev_profit = self.current_profit
        self.current_profit = 0.0

class LandMarket:
    def __init__(self, model):
        self.model = model
        self.developers = [RealEstateDeveloper(model, f"DEV_{i}") for i in range(5)]
        
    def execute_auctions(self):
        """
        Vickrey auction for Cleared (0) land at the urban fringe.
        """
        grid = self.model.grid
        if not hasattr(grid, 'zoning'):
            return
            
        # Find fringe cells (type 0 bordering type 1 or 2)
        fringe_cells = []
        for x in range(grid.width):
            for y in range(grid.height):
                if grid.zoning[x][y] == 0:
                    neighbors = grid.get_neighborhood((x, y), moore=True, include_center=False)
                    if any(grid.zoning[nx][ny] > 0 for nx, ny in neighbors):
                        fringe_cells.append((x, y))
                        
        if not fringe_cells: return
        
        # Auction a random subset of fringe cells
        num_auctions = min(len(fringe_cells), 10)
        auction_cells = np.random.choice(len(fringe_cells), num_auctions, replace=False)
        
        for idx in auction_cells:
            cell = fringe_cells[idx]
            base_price = grid.land_prices[cell[0]][cell[1]]
            
            # Developers submit bids
            bids = []
            for dev in self.developers:
                # Expected future value R_ij + delta_p
                expected_val = base_price * 1.5 
                bid = expected_val * dev.alpha
                if bid <= dev.capital:
                    bids.append((bid, dev))
                    
            if len(bids) >= 2:
                bids.sort(key=lambda x: x[0], reverse=True)
                winner = bids[0][1]
                second_price = bids[1][0]
                
                # Transaction
                winner.capital -= second_price
                grid.zoning[cell[0]][cell[1]] = 1 # Convert to residential
                grid.land_prices[cell[0]][cell[1]] = second_price
                
                # Assume developer immediately builds and sells for expected_val
                winner.capital += expected_val
                winner.current_profit += (expected_val - second_price)
                
    def step(self):
        self.execute_auctions()
        for dev in self.developers:
            dev.adapt_bidding_strategy()
