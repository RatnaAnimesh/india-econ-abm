from mesa.space import MultiGrid
import numpy as np

# Centroids of the 36 states and Union Territories in India (Latitude, Longitude)
STATE_CENTROIDS = {
    "Andhra Pradesh": (15.91, 79.74),
    "Arunachal Pradesh": (28.21, 94.72),
    "Assam": (26.20, 92.93),
    "Bihar": (25.09, 85.31),
    "Chhattisgarh": (21.27, 81.86),
    "Goa": (15.29, 74.12),
    "Gujarat": (22.25, 71.19),
    "Haryana": (29.05, 76.08),
    "Himachal Pradesh": (31.10, 77.17),
    "Jammu & Kashmir": (33.77, 76.57),
    "Jharkhand": (23.61, 85.27),
    "Karnataka": (15.31, 75.71),
    "Kerala": (10.85, 76.27),
    "Madhya Pradesh": (23.64, 78.08),
    "Maharashtra": (19.39, 75.27),
    "Manipur": (24.66, 93.90),
    "Meghalaya": (25.47, 91.36),
    "Mizoram": (23.16, 92.93),
    "Nagaland": (26.15, 94.56),
    "Odisha": (20.53, 84.98),
    "Punjab": (30.91, 75.45),
    "Rajasthan": (26.49, 75.22),
    "Sikkim": (27.53, 88.51),
    "Tamil Nadu": (11.16, 78.76),
    "Telangana": (18.11, 79.02),
    "Tripura": (23.94, 91.98),
    "Uttar Pradesh": (27.07, 80.66),
    "Uttarakhand": (30.06, 79.01),
    "West Bengal": (23.42, 88.15),
    "Delhi": (28.61, 77.20),
    "Chandigarh": (30.73, 76.77),
    "Dadra and Nagar Haveli and Daman and Diu": (20.42, 72.83),
    "Ladakh": (34.15, 77.57),
    "Lakshadweep": (10.57, 72.64),
    "Puducherry": (11.94, 79.80),
    "Andaman & Nicobar Islands": (11.74, 92.65),
    "Andaman and Nicobar Islands": (11.74, 92.65)
}

# Empirical mapping of GSDP, coal shares, NDMA vulnerability, and CBAM exposure
STATE_PROFILES = {
    "Andhra Pradesh": {"gsdp_share": 0.048, "coal_share": 0.50, "vulnerability": 0.35, "cbam_exposure": 0.15},
    "Arunachal Pradesh": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.70, "cbam_exposure": 0.05},
    "Assam": {"gsdp_share": 0.018, "coal_share": 0.20, "vulnerability": 0.80, "cbam_exposure": 0.08},
    "Bihar": {"gsdp_share": 0.032, "coal_share": 0.80, "vulnerability": 0.85, "cbam_exposure": 0.06},
    "Chhattisgarh": {"gsdp_share": 0.020, "coal_share": 0.90, "vulnerability": 0.75, "cbam_exposure": 0.25},
    "Goa": {"gsdp_share": 0.004, "coal_share": 0.10, "vulnerability": 0.20, "cbam_exposure": 0.12},
    "Gujarat": {"gsdp_share": 0.085, "coal_share": 0.65, "vulnerability": 0.28, "cbam_exposure": 0.35},
    "Haryana": {"gsdp_share": 0.038, "coal_share": 0.75, "vulnerability": 0.30, "cbam_exposure": 0.20},
    "Himachal Pradesh": {"gsdp_share": 0.008, "coal_share": 0.00, "vulnerability": 0.45, "cbam_exposure": 0.10},
    "Jammu & Kashmir": {"gsdp_share": 0.008, "coal_share": 0.10, "vulnerability": 0.55, "cbam_exposure": 0.05},
    "Jharkhand": {"gsdp_share": 0.016, "coal_share": 0.95, "vulnerability": 0.90, "cbam_exposure": 0.30},
    "Karnataka": {"gsdp_share": 0.082, "coal_share": 0.30, "vulnerability": 0.32, "cbam_exposure": 0.28},
    "Kerala": {"gsdp_share": 0.038, "coal_share": 0.05, "vulnerability": 0.25, "cbam_exposure": 0.15},
    "Madhya Pradesh": {"gsdp_share": 0.046, "coal_share": 0.75, "vulnerability": 0.60, "cbam_exposure": 0.18},
    "Maharashtra": {"gsdp_share": 0.135, "coal_share": 0.60, "vulnerability": 0.26, "cbam_exposure": 0.32},
    "Manipur": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.65, "cbam_exposure": 0.02},
    "Meghalaya": {"gsdp_share": 0.002, "coal_share": 0.20, "vulnerability": 0.72, "cbam_exposure": 0.04},
    "Mizoram": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.82, "cbam_exposure": 0.02},
    "Nagaland": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.68, "cbam_exposure": 0.02},
    "Odisha": {"gsdp_share": 0.030, "coal_share": 0.85, "vulnerability": 0.78, "cbam_exposure": 0.28},
    "Punjab": {"gsdp_share": 0.028, "coal_share": 0.70, "vulnerability": 0.40, "cbam_exposure": 0.14},
    "Rajasthan": {"gsdp_share": 0.052, "coal_share": 0.70, "vulnerability": 0.55, "cbam_exposure": 0.16},
    "Sikkim": {"gsdp_share": 0.002, "coal_share": 0.00, "vulnerability": 0.42, "cbam_exposure": 0.03},
    "Tamil Nadu": {"gsdp_share": 0.088, "coal_share": 0.45, "vulnerability": 0.28, "cbam_exposure": 0.30},
    "Telangana": {"gsdp_share": 0.048, "coal_share": 0.65, "vulnerability": 0.42, "cbam_exposure": 0.22},
    "Tripura": {"gsdp_share": 0.003, "coal_share": 0.10, "vulnerability": 0.70, "cbam_exposure": 0.02},
    "Uttar Pradesh": {"gsdp_share": 0.082, "coal_share": 0.78, "vulnerability": 0.65, "cbam_exposure": 0.15},
    "Uttarakhand": {"gsdp_share": 0.012, "coal_share": 0.05, "vulnerability": 0.38, "cbam_exposure": 0.08},
    "West Bengal": {"gsdp_share": 0.058, "coal_share": 0.80, "vulnerability": 0.72, "cbam_exposure": 0.20},
    "Delhi": {"gsdp_share": 0.038, "coal_share": 0.40, "vulnerability": 0.35, "cbam_exposure": 0.18},
    "Chandigarh": {"gsdp_share": 0.002, "coal_share": 0.10, "vulnerability": 0.30, "cbam_exposure": 0.10},
    "Dadra and Nagar Haveli and Daman and Diu": {"gsdp_share": 0.003, "coal_share": 0.50, "vulnerability": 0.30, "cbam_exposure": 0.25},
    "Ladakh": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.60, "cbam_exposure": 0.02},
    "Lakshadweep": {"gsdp_share": 0.0001, "coal_share": 0.00, "vulnerability": 0.50, "cbam_exposure": 0.01},
    "Puducherry": {"gsdp_share": 0.002, "coal_share": 0.20, "vulnerability": 0.28, "cbam_exposure": 0.15},
    "Andaman & Nicobar Islands": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.55, "cbam_exposure": 0.02},
    "Andaman and Nicobar Islands": {"gsdp_share": 0.001, "coal_share": 0.00, "vulnerability": 0.55, "cbam_exposure": 0.02}
}

class UrbanSpace(MultiGrid):
    """
    Geographic Space representing Indian states and Union Territories,
    mapping land use zoning, prices, and climate vulnerability indicators.
    """
    def __init__(self, width, height, torus):
        super().__init__(width, height, torus)
        
        # Grid to track zone types
        # 0 = Cleared/Agricultural, 1 = Residential, 2 = Commercial
        self.zoning = np.zeros((width, height), dtype=int)
        
        # Grid to track land prices
        self.land_prices = np.ones((width, height)) * 100.0
        
        # Geographic data layers
        self.state_gsdp_per_capita = np.zeros((width, height))
        self.grid_emission_intensity = np.zeros((width, height))
        self.climate_vulnerability_score = np.zeros((width, height))
        self.cbam_exposure = np.zeros((width, height))
        self.cell_states = {}
        self.cells_by_state = {}
        self.cells_by_state_and_zoning = {}
        
    def coordinate_to_grid(self, lat, lon):
        """Converts geographic coordinates to discrete grid indices."""
        x = int(np.clip((lon - 68.0) / 0.25, 0, self.width - 1))
        y = int(np.clip((lat - 6.5) / 0.25, 0, self.height - 1))
        return (x, y)
        
    def initialize_ntl_proxy(self):
        """
        Maps all grid cells to the closest Indian state centroid
        and initializes geographic profiles, zoning, and land prices.
        """
        centroid_positions = {}
        for state, coords in STATE_CENTROIDS.items():
            centroid_positions[state] = self.coordinate_to_grid(coords[0], coords[1])
            
        self.digital_infrastructure = list(centroid_positions.values())
        
        for x in range(self.width):
            for y in range(self.height):
                # Find nearest state centroid
                nearest_state = min(centroid_positions.keys(), key=lambda s: (centroid_positions[s][0]-x)**2 + (centroid_positions[s][1]-y)**2)
                cx, cy = centroid_positions[nearest_state]
                
                self.cell_states[(x, y)] = nearest_state
                
                # Tag cells with economic and climate vulnerability indicators
                profile = STATE_PROFILES.get(nearest_state, {"gsdp_share": 0.01, "coal_share": 0.50, "vulnerability": 0.50, "cbam_exposure": 0.10})
                self.state_gsdp_per_capita[x][y] = profile["gsdp_share"] * 10.0
                self.grid_emission_intensity[x][y] = 0.2 + 0.8 * profile["coal_share"]
                self.climate_vulnerability_score[x][y] = profile["vulnerability"]
                self.cbam_exposure[x][y] = profile["cbam_exposure"]
                
                # Spatial clustering based on centroid distance
                dist = ((x - cx)**2 + (y - cy)**2)**0.5
                if dist < 4.0:
                    self.zoning[x][y] = 2 # Commercial core
                    self.land_prices[x][y] = 500.0
                elif dist < 12.0:
                    self.zoning[x][y] = 1 # Residential sprawl
                    self.land_prices[x][y] = 200.0
                else:
                    self.zoning[x][y] = 0 # Agricultural/rural
                    self.land_prices[x][y] = 50.0
                    
        # Pre-group cell coordinates by state and zoning for O(1) performance lookup
        self.cells_by_state = {}
        self.cells_by_state_and_zoning = {}
        for pos, st in self.cell_states.items():
            self.cells_by_state.setdefault(st, []).append(pos)
            zone = self.zoning[pos[0]][pos[1]]
            self.cells_by_state_and_zoning.setdefault((st, zone), []).append(pos)
