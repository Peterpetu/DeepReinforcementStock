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
        self.state_size = self.window_size * 10 * 2 + 2  # prices and volumes for window_size days, plus stock_owned and balance
        self.action_size = 3  # buy, sell, hold
    

    def step(self, action):
        if self.current_step >= len(self.stock_price_history):
            self.current_step = 0
        self.current_price = self.stock_price_history[self.current_step, 0, 0]
        assert 0 <= action < self.action_size, f"Invalid action: {action}"
        if self.done:
            return self.state, 0, self.done, {}
        if action == 0:  # Buy
            self.buy_stock()
        elif action == 1:  # Sell
            self.sell_stock()
        elif action == 2:  # Hold
            pass  # Do nothing
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True
            self.current_step = 0  # Reset current_step to 0 when it reaches the size of stock_price_history
        next_state = self.state
        reward = self.balance + self.stock_owned * self.current_price - self.previous_asset_value
        self.previous_asset_value = self.balance + self.stock_owned * self.current_price  # Update previous_asset_value
        return next_state, reward, self.done, {}


    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.cost_basis = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        self.previous_asset_value = self.balance
        # Only include the last n days' data in the state
        self.stock_price_history = self.stock_price_history[-self.window_size:]
        return self.state

    def get_recent_prices_volumes(self, n):
    # Ensure stock_price_history is a 3D array
        assert len(self.stock_price_history.shape) == 3, f"Expected stock_price_history to be a 3D array, got shape {self.stock_price_history.shape}"

    # Get the most recent n days' stock prices and trading volumes
        recent_prices_volumes = self.stock_price_history[-n:, :, :2]  # Get only the first two columns (price and volume) for all windows

    # Ensure the array is of the correct size
        assert recent_prices_volumes.shape == (n, self.stock_price_history.shape[1], 2), f"Expected shape ({n}, {self.stock_price_history.shape[1]}, 2), got {recent_prices_volumes.shape}"


        return recent_prices_volumes


    @property
    def state(self):
        # Get the most recent n days' stock prices and trading volumes
        recent_prices_volumes = self.get_recent_prices_volumes(n=self.window_size)

        # Flatten the recent_prices_volumes array
        recent_prices_volumes = recent_prices_volumes.flatten()

        # Construct the state vector
        state = np.concatenate((recent_prices_volumes, [self.stock_owned, self.balance]))

        return state

    def buy_stock(self):
        shares_to_buy = self.balance // self.current_price
        if shares_to_buy < 1:
            return
        self.stock_owned += shares_to_buy
        self.balance -= shares_to_buy * self.current_price

    def sell_stock(self):
        if self.stock_owned <= 0:
            return
        self.balance += self.stock_owned * self.current_price
        self.stock_owned = 0

