import json
from typing import Dict, List, Tuple
from datamodel import Order, OrderDepth, TradingState, Trade, Observation, Position, Symbol, Time, Listing


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
LIMIT = {
    OSMIUM: 80,
    PEPPER: 80
}

OSMIUM_FAIR = 10_000.9
OSMIUM_EDGE = 7
OSMIUM_EDGE_FAR = 12
OSMIUM_SIZE_NEAR = 60
OSMIUM_SIZE_FAR = 20
OSMIUM_SKEW_DENOM = 40
OSMIUM_MICRO_CAP = 0.5
OSMIUM_TAKE_SLOP = 1
#----------------------
PEPPER_FAIR = 12500
PEPPER_FAST_ALPHA = 0.35
PEPPER_SLOW_ALPHA = 0.05
PEPPER_SLOPE_ALPHA = 0.04
PEPPER_PRIOR_SLOPE = 0.0
PEPPER_FORECAST_TS_BASE = 800
PEPPER_FORECAST_TS_MAX = 1500
PEPPER_BID_EDGE_BASE = 3
PEPPER_ASK_EDGE_BASE = 4
PEPPER_SKEW_DENOM = 16
class Trader:
    def __init__(self, state: TradingState):
        
        pass
    def get_position(self, product:str, state: TradingState) -> int:
        return state.position.get(product, 0)
    
    def get_best_bid(self, order_depth: OrderDepth, state: TradingState) -> int:
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        # This is the best bid price in the order book
    def get_best_ask(self, order_depth: OrderDepth, state: TradingState) -> int:
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        # This is the best ask price in the order book
        
    def get_mid(self, best_bid: int, best_ask: int) -> float:
        return (best_bid + best_ask) / 2.0
    
    def get_fair(self, product:str, state: TradingState) -> float:
        if product == OSMIUM:
            return OSMIUM_FAIR
        elif product == PEPPER:
            return PEPPER_FAIR
        else:
            return 0
    def get_ema(self, product:str, state: TradingState) -> float:
        if product == OSMIUM:
            
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        for product, order_depth in state.order_depths.items():
            position = self.get_position(product, state)
            
            
        
        
    
    
    
    
    