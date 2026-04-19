import json
from typing import Dict, List, Tuple
from datamodel import Order, OrderDepth, TradingState, Trade, Observation, Position, Symbol, Time, Listing


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
LIMIT = {
    OSMIUM: 80,
    PEPPER: 80
}

class Trader:
    def __init__(self, state: TradingState):
        self.EMA: list[float] = []
        self.alpha: float = 0.02
        self.midPrice: list[float] = []
        self.orders: list[Order] = []
        
        
    
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
            pass
    def total(self, state: TradingState, symbol: str) -> List[Order]:
        orders: List[Order] = []
        if symbol == PEPPER:
            orders = self.pepper_Strategy(state, symbol)
        elif symbol == OSMIUM:
            orders = self.osmium_Strategy(state, symbol)
        return orders
    def microPrice(self, state: TradingState, symbol: str) -> float:
        bid = state.order_depths[symbol].buy_orders
        ask = state.order_depths[symbol].sell_orders
        best_bid = max(bid.keys())
        best_ask = min(ask.keys())
        depth = state.order_depths[symbol]
        bid_volume = sum(depth.buy_orders.values())
        ask_volume = sum(depth.sell_orders.values())
        if bid_volume > 0 and ask_volume > 0:
            return (bid_volume * best_bid + ask_volume * best_ask) / (bid_volume + ask_volume)
        elif bid_volume > 0:
            return best_bid
        elif ask_volume > 0:
            return best_ask
        else:
            return 0
    def EMA(self, price: float, alpha: float) -> float:
        return alpha * price + (1 - alpha) * self.EMA(price, alpha)

    def pepper_Strategy(self, state: TradingState, symbol: str) -> List[Order]:
        bid = state.order_depths[symbol].buy_orders
        ask = state.order_depths[symbol].sell_orders
        best_bid = max(bid.keys())
        best_ask = min(ask.keys())
        mid = (best_bid + best_ask) / 2.0
        
        
            
        
            
    
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        self.orders = []
        for product, order_depth in state.order_depths.items():
            position = self.get_position(product, state)
            order_buy = state.order_depths[product].buy_orders
            order_sell = state.order_depths[product].sell_orders
            best_bid = max(order_buy.keys())
            pass
            
        
        
    
    
    
    
    