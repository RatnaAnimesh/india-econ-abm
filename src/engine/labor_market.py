import numpy as np

class LaborMarket:
    """
    Decentralized Labor Market with Search and Matching Frictions.
    Implements gravity model matching and hysteresis.
    """
    def __init__(self, model):
        self.model = model
        self.phi = 0.5 # Wage elasticity in matching
        self.gamma_v = 1.0 # Vacancy elasticity
        self.gamma_s = 1.0 # Skill elasticity
        self.gamma_d = 1.0 # Distance penalty
        
        # Hysteresis parameters
        self.xi = 0.05 # Unemployment decay
        self.zeta = 0.1 # Employment upward adjustment
        
    def match_workers(self):
        """
        Executes the matching protocol between searching firms and unemployed households.
        """
        # Collect vacancies from firms
        # A firm wants to hire if its target production exceeds its labor capacity
        # For simplicity, if actual sales > max capacity, open vacancies
        searching_firms = []
        for firm in self.model.agents:
            if hasattr(firm, 'expected_demand') and hasattr(firm, 'inventory'):
                target_prod = firm.expected_demand * (1.0 + firm.buffer_ratio) - firm.inventory
                current_cap = firm.tfp * (firm.capital ** firm.cap_share) * (firm.labor ** firm.lab_share)
                
                if target_prod > current_cap:
                    # Estimate needed labor (inverted Cobb-Douglas approximation)
                    needed_ratio = target_prod / max(0.01, current_cap)
                    desired_labor = int(firm.labor * needed_ratio)
                    vacancies = desired_labor - firm.labor
                    if vacancies > 0:
                        searching_firms.append({
                            'firm': firm,
                            'vacancies': vacancies,
                            'wage_offer': firm.wage_rate * self.model.price_level
                        })
                        
        unemployed_households = [hh for hh in self.model.households if not hh.employed]
        
        # Spatial Gravity Matching
        # In a highly populated simulation, doing O(N*M) is too slow.
        # We sample a subset of firms for each worker.
        for hh in unemployed_households:
            if not searching_firms:
                break # No jobs left
                
            # Randomly sample up to 5 firms to evaluate (Network search proxy)
            sample_size = min(5, len(searching_firms))
            sampled_firms = np.random.choice(searching_firms, sample_size, replace=False)
            
            best_match = None
            best_prob = 0.0
            
            for firm_data in sampled_firms:
                firm = firm_data['firm']
                # Distance proxy: 1.0 if same state, 5.0 if different
                distance = 1.0 if (hasattr(hh, 'state') and hasattr(firm, 'state') and hh.state == firm.state) else 5.0
                
                # Gravity equation
                prob = ((firm_data['vacancies'] ** self.gamma_v) * (hh.skill_level ** self.gamma_s)) / (distance ** self.gamma_d)
                wage_gap = max(0, hh.reservation_wage - firm_data['wage_offer'])
                prob *= np.exp(-self.phi * wage_gap)
                
                if prob > best_prob and firm_data['wage_offer'] >= hh.reservation_wage:
                    best_prob = prob
                    best_match = firm_data
                    
            if best_match is not None:
                # Consummate match
                hh.employed = True
                hh.employer = best_match['firm']
                hh.wage = best_match['wage_offer']
                best_match['firm'].labor += 1
                best_match['vacancies'] -= 1
                
                if best_match['vacancies'] <= 0:
                    searching_firms.remove(best_match)
                    
    def update_reservation_wages(self):
        """Applies hysteresis to household reservation wages."""
        for hh in self.model.households:
            if hh.employed:
                hh.reservation_wage += self.zeta * (hh.wage - hh.reservation_wage)
            else:
                # Psychological decay due to prolonged unemployment
                hh.reservation_wage *= (1.0 - self.xi)
                
    def step(self):
        self.match_workers()
        self.update_reservation_wages()
