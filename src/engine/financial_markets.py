import numpy as np
from mesa import Agent

class BehavioralTrader(Agent):
    """
    Financial trader that participates in the Limit Order Book.
    """
    def __init__(self, model, unique_id, strategy):
        super().__init__(model)
        self.strategy = strategy # 'Fundamentalist', 'Overconfident', 'Herder', 'Disposition'
        self.cash = 10000.0
        self.holdings = 100
        self.reference_price = 100.0 # Purchase price for disposition effect
        
    def generate_order(self, current_price, fundamental_value, market_trend):
        """Generates a bid or ask based on behavioral strategy."""
        bid, ask = None, None
        
        if self.strategy == 'Fundamentalist':
            # Buy if undervalued, sell if overvalued
            if current_price < fundamental_value * 0.95:
                bid = current_price * 1.01
            elif current_price > fundamental_value * 1.05 and self.holdings > 0:
                ask = current_price * 0.99
                
        elif self.strategy == 'Overconfident':
            # Trades excessively with perceived signals (noise)
            signal = np.random.normal(0, 5) # High variance signal
            if signal > 2:
                bid = current_price * 1.02
            elif signal < -2 and self.holdings > 0:
                ask = current_price * 0.98
                
        elif self.strategy == 'Herder':
            # Follows the recent market trend
            if market_trend > 0:
                bid = current_price * 1.01 # Buying momentum
            elif market_trend < 0 and self.holdings > 0:
                ask = current_price * 0.99 # Panic selling
                
        elif self.strategy == 'Disposition':
            # Sells winners too early, holds losers too long
            profit_margin = (current_price - self.reference_price) / self.reference_price
            if profit_margin > 0.05 and self.holdings > 0:
                ask = current_price # Lock in gain
            elif profit_margin < -0.1:
                # Double down on loser
                bid = current_price
                
        return bid, ask


class LimitOrderBook:
    """
    Continuous Double Auction mechanism for trading a proxy market index.
    """
    def __init__(self, model):
        self.model = model
        self.bids = [] # (price, agent)
        self.asks = [] # (price, agent)
        self.current_price = 100.0
        self.price_history = [100.0]
        
        strategies = ['Fundamentalist', 'Overconfident', 'Herder', 'Disposition']
        self.traders = [BehavioralTrader(model, f"TRADER_{i}", np.random.choice(strategies)) for i in range(20)]
        
    def step(self):
        from src.engine.model import FirmAgent
        # Calculate fundamental value proxy based on total economy firm profit
        total_profit = sum([max(0, a.profit) for a in self.model.agents if isinstance(a, FirmAgent)])
        fundamental_value = 100.0 + (total_profit / 10000.0) # Arbitrary scaling
        
        market_trend = self.price_history[-1] - self.price_history[-2] if len(self.price_history) > 1 else 0
        
        self.bids = []
        self.asks = []
        
        # Collect orders
        for trader in self.traders:
            bid, ask = trader.generate_order(self.current_price, fundamental_value, market_trend)
            if bid and trader.cash >= bid:
                self.bids.append((bid, trader))
            if ask and trader.holdings > 0:
                self.asks.append((ask, trader))
                
        # Sort LOB
        self.bids.sort(key=lambda x: x[0], reverse=True) # Highest bid first
        self.asks.sort(key=lambda x: x[0])               # Lowest ask first
        
        # Match orders
        transactions = []
        while self.bids and self.asks:
            best_bid, buyer = self.bids[0]
            best_ask, seller = self.asks[0]
            
            if best_bid >= best_ask:
                # Match! Execution price is the midpoint
                exec_price = (best_bid + best_ask) / 2.0
                
                buyer.cash -= exec_price
                buyer.holdings += 1
                buyer.reference_price = exec_price # Reset reference
                
                seller.cash += exec_price
                seller.holdings -= 1
                
                transactions.append(exec_price)
                
                self.bids.pop(0)
                self.asks.pop(0)
            else:
                break # No more crossed orders
                
        if transactions:
            self.current_price = np.mean(transactions)
            self.price_history.append(self.current_price)
