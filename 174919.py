import json

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import numpy as np
import string

class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80
    }

    def bid(self):
        return 15

    '''
    ACO shows volatility, but seems to always revert back to a mean price of about 10000. We will implement a mean reversion strategy, 
    where we will place buy orders when the price is significantly below the mean price and sell orders when the price is significantly
    above the mean price. When the price reverts back to the mean, we will exit our long or short position. For now, we will maintain
    a rolling window of the last 20 mid prices and use the average of those mid prices as our mean price (may test a hard mean of 
    10000). We will also check the position limits before placing any orders.
    '''
    def aco_strategy(self, state: TradingState):
        orders: List[Order] = []
        sell_orders = state.order_depths["ASH_COATED_OSMIUM"].sell_orders.items()
        buy_orders = state.order_depths["ASH_COATED_OSMIUM"].buy_orders.items()

        try:
            traderData = json.loads(state.traderData)
        except:
            traderData = {}
        
        if "ACO" not in traderData:
            traderData["ACO"] = {}
            traderData["ACO"]["WINDOW"] = []
        
        rolling_window = traderData["ACO"]["WINDOW"]

        if len(sell_orders) == 0 or len(buy_orders) == 0:
            return orders, rolling_window

        position = state.position.get("ASH_COATED_OSMIUM", 0)
        LIMIT = self.POSITION_LIMITS["ASH_COATED_OSMIUM"]

        best_ask = min(sell_orders, key=lambda x: x[0])[0]
        best_bid = max(buy_orders, key=lambda x: x[0])[0]
        mid_price = (best_ask + best_bid) / 2

        print(f"Current position for ACO: {position}, Best bid: {best_bid}, Best ask: {best_ask}, Mid price: {mid_price}")

        if len(rolling_window) == 20:
            rolling_mean = sum(rolling_window) / len(rolling_window)
            stdev = np.std(rolling_window)
            z_score = (mid_price - rolling_mean) / stdev

            if position > 0 and z_score >= -0.5:
                orders.append(Order("ASH_COATED_OSMIUM", best_ask - 1, -position))
                print(f"Exiting long position with {position} shares at price {best_ask - 1} with z-score {z_score}")
                position = 0
                return orders, rolling_window

            if position < 0 and z_score <= 0.5:
                orders.append(Order("ASH_COATED_OSMIUM", best_bid + 1, -position))
                print(f"Exiting short position with {-position} shares at price {best_bid + 1} with z-score {z_score}")
                position = 0
                return orders, rolling_window
            
            if z_score <= -2 and LIMIT >= position:
                buy_amount = min(40, LIMIT - position)
                orders.append(Order("ASH_COATED_OSMIUM", best_bid + 1, buy_amount))
                print(f"Placing buy order for {buy_amount} shares at price {best_bid + 1} with z-score {z_score}")
                position += buy_amount
            elif z_score <= -1.75 and LIMIT >= position:
                buy_amount = min(30, LIMIT - position)
                orders.append(Order("ASH_COATED_OSMIUM", best_bid + 1, buy_amount))
                print(f"Placing buy order for {buy_amount} shares at price {best_bid + 1} with z-score {z_score}")
                position += buy_amount
            elif z_score <= -1.5 and LIMIT >= position:
                buy_amount = min(20, LIMIT - position)
                orders.append(Order("ASH_COATED_OSMIUM", best_bid + 1, buy_amount))
                print(f"Placing buy order for {buy_amount} shares at price {best_bid + 1} with z-score {z_score}")
                position += buy_amount
            elif z_score >= 2 and position > -LIMIT:
                sell_amount = min(40, LIMIT + position)
                orders.append(Order("ASH_COATED_OSMIUM", best_ask - 1, -sell_amount))
                print(f"Placing sell order for {sell_amount} shares at price {best_ask - 1} with z-score {z_score}")
                position -= sell_amount
            elif z_score >= 1.75 and position > -LIMIT:
                sell_amount = min(30, LIMIT + position)
                orders.append(Order("ASH_COATED_OSMIUM", best_ask - 1, -sell_amount))
                print(f"Placing sell order for {sell_amount} shares at price {best_ask - 1} with z-score {z_score}")
                position -= sell_amount
            elif z_score >= 1.5 and position > -LIMIT:
                sell_amount = min(20, LIMIT + position)
                orders.append(Order("ASH_COATED_OSMIUM", best_ask - 1, -sell_amount))
                print(f"Placing sell order for {sell_amount} shares at price {best_ask - 1} with z-score {z_score}")
                position -= sell_amount

        rolling_window.append(mid_price)

        if len(rolling_window) > 20:
            rolling_window.pop(0)

        return orders, rolling_window

    '''
    IPR shows a continual increase in price. We will implement a simple market making strategy, where we will place buy orders better
    than the current highest bid in the book and sell orders worse than the current lowest ask. We will also check the position limits 
    before placing any orders.
    '''
    def ipr_strategy(self, state: TradingState):
        orders: List[Order] = []
        sell_orders = state.order_depths["INTARIAN_PEPPER_ROOT"].sell_orders.items()
        buy_orders = state.order_depths["INTARIAN_PEPPER_ROOT"].buy_orders.items()

        if len(sell_orders) == 0 or len(buy_orders) == 0:
            return orders

        position = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        LIMIT = self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]

        lowest_ask = min(sell_orders, key=lambda x: x[0])[0]
        highest_bid = max(buy_orders, key=lambda x: x[0])[0]
        mid_price = (lowest_ask + highest_bid) / 2

        print(f"Current position for IPR: {position}, Best bid: {highest_bid}, Best ask: {lowest_ask}, Mid price: {mid_price}")

        # Search thru sell orders and place buy orders lower than the mid price, while ensuring we do not exceed position limits
        for price, amount in sell_orders:
            if price < mid_price and position < LIMIT:
                buy_amount = min(-amount, LIMIT - position)
                orders.append(Order("INTARIAN_PEPPER_ROOT", price, buy_amount))
                print(f"Placing buy order for {buy_amount} shares at price {price} from order book")
                position += buy_amount

        # Search thru buy orders and place sell orders higher than the mid price, while ensuring we do not exceed position limits
        for price, amount in buy_orders:
            if price > mid_price and position > -LIMIT:
                sell_amount = min(amount, LIMIT + position)
                orders.append(Order("INTARIAN_PEPPER_ROOT", price, -sell_amount))
                print(f"Placing sell order for {sell_amount} shares at price {price} from order book")
                position -= sell_amount
        
        # Market making: place buy order one tick above the best bid and sell order one tick below the best ask, while ensuring we do not exceed position limits
        bid_price = highest_bid + 1
        bid_amount = LIMIT - position
        sell_price = lowest_ask - 1
        sell_amount = LIMIT + position

        if bid_price < mid_price and sell_price > mid_price and bid_amount > 0 and sell_amount > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, bid_amount))
            orders.append(Order("INTARIAN_PEPPER_ROOT", sell_price, -sell_amount))
            print(f"Placing market making buy order for {bid_amount} shares at price {bid_price}")
            print(f"Placing market making sell order for {sell_amount} shares at price {sell_price}")

        return orders
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        try:
            traderData = json.loads(state.traderData)
        except:
            traderData = {}

        rolling_window_aco = []
        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            orders = []
            if product == "ASH_COATED_OSMIUM":
                orders, rolling_window_aco = self.aco_strategy(state)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders = self.ipr_strategy(state)
            
            result[product] = orders

        if "ACO" not in traderData:
            traderData["ACO"] = {}
            traderData["ACO"]["WINDOW"] = []
        
        if rolling_window_aco:
            traderData["ACO"]["WINDOW"] = rolling_window_aco

        traderData = json.dumps(traderData)
        conversions = 0
        return result, conversions, traderData