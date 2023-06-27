import numpy as np
class Environment:
    def __init__(self, stock_price_history, initial_balance=10000, window_size=30):
        self.stock_price_history = np.array(stock_price_history)
        self.n_steps = len(self.stock_price_history)
        self.current_step = 0
        self.stock_owned = 0
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.done = False
        self.window_size = window_size
        self.state_size = self.window_size * 2 + 2  # prices and volumes for window_size days, plus stock_owned and balance
        self.action_size = 2  # buy, sell

    def step(self, action):
        assert self.action_space.contains(action)
        if self.done:
            return self.state, 0, self.done, {}
        if action == 0:  # Buy
            self.buy_stock()
        elif action == 1:  # Sell
            self.sell_stock()
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True
        next_state = self.state
        reward = self.balance + self.stock_owned * self.current_price - self.previous_asset_value
        self.previous_asset_value = self.balance + self.stock_owned * self.current_price
        return next_state, reward, self.done, {}

    def reset(self):
        self.stock_owned = 0
        self.balance = self.initial_balance
        self.done = False
        self.current_step = 0
        return self.state
    
    def get_recent_prices_volumes(self, n):
    # Get the most recent n days' stock prices and trading volumes
        recent_prices_volumes = self.stock_price_history[-n:]
    
    # Flatten the array into a one-dimensional array
        recent_prices_volumes = recent_prices_volumes.flatten()
    
        return recent_prices_volumes


    @property
    def state(self):
    # Get the most recent n days' stock prices and trading volumes
        recent_prices_volumes = self.get_recent_prices_volumes(n=self.window_size)
    
    # Construct the state vector
        state = np.concatenate((recent_prices_volumes, [self.stock_owned, self.balance]))
    
        return state


    @property
    def current_price(self):
        return self.stock_price_history[self.current_step]

    def buy_stock(self):
        shares_to_buy = self.balance // self.current_price
        if shares_to_buy < 1:
            return
        self.stock_owned += shares_to_buy
        self.balance -= shares_to_buy * self.current_price

    def sell_stock(self):
        self.balance += self.stock_owned * self.current_price
        self.stock_owned = 0
