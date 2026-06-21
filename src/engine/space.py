from mesa.space import MultiGrid
import numpy as np

class UrbanSpace(MultiGrid):
    """
    Coordinate grid tracking Land Use: Cleared (l), Residential (r), Commercial (c).
    """
    def __init__(self, width, height, torus):
        super().__init__(width, height, torus)
        
        # Grid to track zone types
        # 0 = Cleared/Agricultural, 1 = Residential, 2 = Commercial
        self.zoning = np.zeros((width, height), dtype=int)
        
        # Grid to track land prices
        self.land_prices = np.ones((width, height)) * 100.0 # Base price
        
    def initialize_ntl_proxy(self):
        """
        Uses a proxy for Nighttime Lights (NTL) to seed initial urbanization clusters.
        """
        # Create a few random urban centers
        num_centers = 5
        centers = [(np.random.randint(self.width), np.random.randint(self.height)) for _ in range(num_centers)]
        
        # Infrastructure deployed at urban centers
        self.digital_infrastructure = centers.copy()
        
        for x in range(self.width):
            for y in range(self.height):
                # Calculate distance to nearest center
                min_dist = min([((x - cx)**2 + (y - cy)**2)**0.5 for cx, cy in centers])
                
                if min_dist < 5:
                    self.zoning[x][y] = 2 # Commercial core
                    self.land_prices[x][y] = 500.0
                elif min_dist < 15:
                    self.zoning[x][y] = 1 # Residential sprawl
                    self.land_prices[x][y] = 200.0
                else:
                    self.zoning[x][y] = 0 # Cleared/Agri
                    self.land_prices[x][y] = 50.0
