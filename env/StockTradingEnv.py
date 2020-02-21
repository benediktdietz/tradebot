import random
import json
import gym
import sys
from gym import spaces
import pandas as pd
import numpy as np
from scipy.special import softmax


MAX_ACCOUNT_BALANCE = 1e+10
MAX_NUM_SHARES = 1e+8
MAX_SHARE_PRICE = 10000
MAX_OPEN_POSITIONS = 200
MAX_STEPS = 120

NUM_PAST_DATA_POINTS = 10
NUM_FEATURES = 5

TRANSACTION_FEE = 0.01

INITIAL_ACCOUNT_BALANCE = 10000





def build_obs(dataframe, step, num_symbols, balance, shares_held):

    feat_keys = ['Open', 'Close', 'High', 'Low', 'Volume']
    # feat_keys = ['High', 'Low', 'Volume']
    new_feats = ['hist_open',               # 0
                 'hist_close',              # 1
                 'hist_low',                # 2
                 'hist_high',               # 3
                 'hist_avg',                # 4
                 'hist_variance',           # 5
                 'hist_median',             # 6
                 'hist_volume_avg',         # 7
                 'hist_volume_variance',    # 8
                 'hist_volume_median',      # 9
                 'Volume',                  # 10
                 'current_est_price',       # 11
                 'shares_held',             # 12
                 'shares_held_value',       # 13
                 'balance'                  # 14
                 ]
    


    obs_array = np.zeros((NUM_PAST_DATA_POINTS, num_symbols, len(feat_keys)))

    obs_array_new = np.zeros((num_symbols, len(new_feats)))


    if obs_array.shape[0] != dataframe.iloc[step - NUM_PAST_DATA_POINTS : step][feat_keys[0]].values.shape[0]:
        print('\ndimension error at step ', step)

    for feat in range(len(feat_keys)):

        # obs_array[:,:,feat] = dataframe.iloc[step - NUM_PAST_DATA_POINTS : step][feat_keys[feat]].values / np.amax(dataframe.iloc[0 : -1][feat_keys[feat]].values)
        obs_array[:,:,feat] = dataframe.iloc[step - NUM_PAST_DATA_POINTS : step][feat_keys[feat]].values

    price_array = make_price_array(obs_array, num_symbols)

    obs_array_new[:,0] = price_array[0,:]
    obs_array_new[:,1] = price_array[-1,:]
    obs_array_new[:,2] = np.amin(price_array, axis=0)
    obs_array_new[:,3] = np.amax(price_array, axis=0)
    obs_array_new[:,4] = np.mean(price_array, axis=0)
    obs_array_new[:,5] = np.var(price_array, axis=0)
    obs_array_new[:,6] = np.median(price_array, axis=0)

    obs_array_new[:,7] = np.mean(obs_array[:, :, -1])
    obs_array_new[:,8] = np.var(obs_array[:, :, -1])
    obs_array_new[:,9] = np.median(obs_array[:, :, -1])

    obs_array_new[:,10] = obs_array[-1, :, -1]

    obs_array_new[:,11] = get_current_prices(dataframe, step)

    obs_array_new[:,12] = shares_held

    obs_array_new[:,13] = shares_held * obs_array_new[:,11]

    obs_array_new[:,14] = balance


    obs_array_new = np.reshape(obs_array_new, [-1, 1])

    obs_array_new[obs_array_new == np.nan] = 0.


    # # print('\n-------> ', obs_array.shape)
    # obs_array = np.append(obs_array, add_obs_vecs)
    # # print('\n-------> ', obs_array.shape)
    # obs_array = np.append(obs_array, add_obs_singles)
    # # print('\n-------> ', obs_array.shape)
    # # print()

    return np.asarray(obs_array_new, dtype=np.float32)


def make_price_array(obs_array, num_symbols):
   
    feat_hist_open = np.asarray(obs_array[:, :, 0], dtype=np.float16)
    feat_hist_close = np.asarray(obs_array[:, :, 1], dtype=np.float16)


    dummy_price_array = np.zeros((NUM_PAST_DATA_POINTS, num_symbols))

    def random_price(a, b):
        # print(a,b)
        try:
            return(np.random.uniform(a, b))
        except OverflowError:
            print('\n', 'NaN error @ make_price_array function  | inputs: ', a, b, '\n')

    random_price_vectorizer = np.vectorize(random_price)

    for j in range(num_symbols):
        # print(j)
        # print(random_price_vectorizer(feat_hist_open[:,j], feat_hist_close[:,j]).shape)
        dummy_price_array[:,j] = random_price_vectorizer(feat_hist_open[:,j], feat_hist_close[:,j])

    return dummy_price_array


def get_current_prices(dataframe, step):

    def rand_uni(a, b):
        return(np.random.uniform(a, b))
    
    rand_price_vectorizer = np.vectorize(rand_uni)

    open_prices = np.asarray(dataframe.iloc[step]['Open'].values, dtype=np.float16)
    close_prices = np.asarray(dataframe.iloc[step]['Close'].values, dtype=np.float16)

    if np.nan in open_prices or np.nan in close_prices:
        print('\n\nNaNs@get_current_prices!\n\n')

    try:
        return rand_price_vectorizer(open_prices, close_prices)
    except OverflowError:
        print('\n', 'NaN error @ get_current_prices function  | inputs:\n', open_prices, '\n*******\n', close_prices, '\n')





class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, num_symbols):
        super(StockTradingEnv, self).__init__()

        self.df = df

        self.normalized_df = (self.df - self.df.min()) / (self.df.max() - self.df.min() + 1e-4)

        self.normalized_df.fillna(method ='pad') 
        self.normalized_df.fillna(0.) 
        self.normalized_df.replace(to_replace = np.nan, value = 0.)


        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.num_symbols = num_symbols
        self.num_features = 15


        # self.action_space = spaces.Box(
        #     low = np.array([0, 0, 0]), 
        #     high = np.array([3, 1, self.num_symbols]), dtype = np.float16)


        self.action_space = spaces.Box(
            low = np.zeros(self.num_symbols + 3), 
            high = np.ones(self.num_symbols + 3), 
            dtype = np.float16)


        # # # symbol index x num_features x num_past_data
        # lows = np.zeros(NUM_PAST_DATA_POINTS * self.num_features * self.num_symbols + 3 * self.num_symbols + 2, dtype=np.float16)
        # highs = np.inf * np.ones(NUM_PAST_DATA_POINTS * self.num_features * self.num_symbols + 3 * self.num_symbols + 2, dtype=np.float16)
        # self.observation_space = spaces.Box(lows, highs, dtype=np.float16)

        # # symbol index x num_features x num_past_data
        lows = np.zeros(self.num_features * self.num_symbols, dtype=np.float16)
        highs = np.ones(self.num_features * self.num_symbols, dtype=np.float16)
        self.observation_space = spaces.Box(lows, highs, dtype=np.float16)


    def _next_observation(self):

        # try:
        #     obs_dummy = build_obs(self.normalized_df, self.current_step, self.num_symbols, self.balance, self.shares_held)
        # except OverflowError:
        #     print('\n\n\nerror at step ', self.current_step, '\n\n\n')

        obs_dummy = build_obs(self.normalized_df, self.current_step, self.num_symbols, self.balance, self.shares_held)

        return obs_dummy


    def _take_action(self, action):
        # Set the current price to a random price within the time step
     
        self.current_price = get_current_prices(self.df, self.current_step)

        # self.action_type = int(np.abs(np.round(action[0]))) #int
        # self.amount = np.abs(action[1]) #float
        # self.action_sym = int(np.abs(np.round(action[2]))) #int

        self.action_sym = np.argmax(softmax(action[:self.num_symbols]))
        self.amount = softmax(action[:self.num_symbols])[self.action_sym]
        self.action_type = np.argmax(softmax(action[self.num_symbols:]))

        # if self.amount > 1.:
        #     self.amount = 1.

        # if self.action_sym >= self.num_symbols:
        #     self.action_sym = np.random.randint(0, self.num_symbols)


        if self.action_type == 0:
            # Buy self.amount % of balance in shares
            total_possible = np.floor(self.balance / self.current_price[self.action_sym])

            if total_possible >= 1.:
                shares_bought = int(total_possible * self.amount)
                # prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * self.current_price[self.action_sym]
                transaction_cost = additional_cost * TRANSACTION_FEE

                self.balance -= additional_cost
                self.balance -= transaction_cost
                # self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held[self.action_sym] += shares_bought

                self.total_shares_purchased[self.action_sym] += shares_bought
                self.total_fees += transaction_cost


        elif self.action_type == 1:

            if int(np.round(self.shares_held[self.action_sym])) > 0:

                # Sell self.amount % of shares held
                shares_sold = int(np.round(int(np.round(self.shares_held[self.action_sym])) * self.amount))

                self.balance += shares_sold * self.current_price[self.action_sym]

                transaction_cost = (shares_sold * self.current_price[self.action_sym]) * TRANSACTION_FEE

                self.balance -= transaction_cost

                self.shares_held[self.action_sym] -= shares_sold
                self.total_shares_sold[self.action_sym] += shares_sold
                self.total_sales_value[self.action_sym] += shares_sold * self.current_price[self.action_sym]

                self.total_fees += transaction_cost


        self.net_worth = self.balance + np.sum(self.shares_held * self.current_price)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth


    def _take_action_old(self, action):
        # Set the current price to a random price within the time step
     
        self.current_price = get_current_prices(self.df, self.current_step)

        self.action_type = int(np.abs(np.round(action[0]))) #int
        self.amount = np.abs(action[1]) #float
        self.action_sym = int(np.abs(np.round(action[2]))) #int

        if self.amount > 1.:
            self.amount = 1.

        if self.action_sym >= self.num_symbols:
            self.action_sym = np.random.randint(0, self.num_symbols)


        if self.action_type == 0:
            # Buy self.amount % of balance in shares
            total_possible = np.floor(self.balance / self.current_price[self.action_sym])

            if total_possible >= 1.:
                shares_bought = int(total_possible * self.amount)
                # prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * self.current_price[self.action_sym]
                transaction_cost = additional_cost * TRANSACTION_FEE

                self.balance -= additional_cost
                self.balance -= transaction_cost
                # self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held[self.action_sym] += shares_bought

                self.total_shares_purchased[self.action_sym] += shares_bought
                self.total_fees += transaction_cost


        elif self.action_type == 1:

            if int(np.round(self.shares_held[self.action_sym])) > 0:

                # Sell self.amount % of shares held
                shares_sold = int(np.round(int(np.round(self.shares_held[self.action_sym])) * self.amount))

                self.balance += shares_sold * self.current_price[self.action_sym]

                transaction_cost = (shares_sold * self.current_price[self.action_sym]) * TRANSACTION_FEE

                self.balance -= transaction_cost

                self.shares_held[self.action_sym] -= shares_sold
                self.total_shares_sold[self.action_sym] += shares_sold
                self.total_sales_value[self.action_sym] += shares_sold * self.current_price[self.action_sym]

                self.total_fees += transaction_cost


        self.net_worth = self.balance + np.sum(self.shares_held * self.current_price)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df) - 1:
            self.current_step = NUM_PAST_DATA_POINTS + 1

        if self.current_step < NUM_PAST_DATA_POINTS:
            self.current_step = NUM_PAST_DATA_POINTS + 1


        delay_modifier = (self.current_step / MAX_STEPS)

        # reward = (2 * self.balance * delay_modifier) + np.sum(self.shares_held * self.current_price)
        # if self.action_type not in [0, 1]:
        #     reward *= .8
        # reward = np.sum(self.shares_held * self.current_price) + self.balance - INITIAL_ACCOUNT_BALANCE
        reward = np.sum(self.shares_held * self.current_price) + self.balance 
        reward /= INITIAL_ACCOUNT_BALANCE
        reward *= 100.
        if int(self.action_type) != 0 and np.sum(self.shares_held) == 0:
            reward -= np.abs(.5 * reward)
        if int(self.action_type) == 1 and self.shares_held[self.action_sym] == 0:
            reward -= np.abs(1. * reward)


        done = self.net_worth <= 0 or self.current_step > MAX_STEPS or self.current_step > (len(self.df) - 1)

        obs = self._next_observation()

        monitor_data = {
                'balance': self.balance,
                'net_worth': self.net_worth,
                'total_fees': self.total_fees,
                'reward': reward,
                'shares_held': self.shares_held,
                'total_shares_purchased': self.total_shares_purchased,
                'total_shares_sold': self.total_shares_sold,
                'total_sales_value': self.total_sales_value,
                'action_sym': self.action_sym,
                'action_amount': self.amount,
                'action_type': self.action_type,
                }

        return obs, reward, done, {}, monitor_data


    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = np.zeros(self.num_symbols, dtype=np.float64)
        self.total_fees = 0.0
        # self.cost_basis = 0
        self.total_shares_purchased = np.zeros(self.num_symbols, dtype=np.float64)
        self.total_shares_sold = np.zeros(self.num_symbols, dtype=np.float64)
        self.total_sales_value = np.zeros(self.num_symbols, dtype=np.float64)

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(NUM_PAST_DATA_POINTS + 1, len(self.df) - 1)

        return self._next_observation()


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        if self.current_step == 400:
            # print(f'Step: {self.current_step}')
            # print(f'Balance: {np.round(self.balance, 1)}')
            # print(
            #     f'Shares held: {int(np.sum(self.shares_held))} (Total sold: {int(np.sum(self.total_shares_sold))})')
            # print(
            #     f'Total sales value: {np.round(np.sum(self.total_sales_value), 1)}')
            # print(
            #     f'Net worth: {np.round(self.net_worth, 1)}')
            # print(f'Profit: {np.round(profit, 1)}\n')

            print('Step: ', int(self.current_step), 
                # '\nAction--->  type/amount/symbol ', self.action_type, '/ ', self.amount, '/ ', symbol,
                '\nBalance: ', np.round(self.balance, 1), 
                '\nShares held/purch./sold: ', int(np.sum(self.shares_held)), '/ ', int(np.sum(self.total_shares_purchased)), '/ ', int(np.sum(self.total_shares_sold)), 
                '\nFees: ', np.round(self.total_fees, 1),
                '\nProfit: ', np.round(profit, 1),
                end = '\n\r\n')


    def _render_to_file(self, filename='render.txt'):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        file = open(filename, 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {int(np.sum(self.shares_held))} (Total sold: {int(np.sum(self.total_shares_sold))})\n')
        file.write(f'Net worth: {self.net_worth}\n')
        file.write(f'Profit: {profit}\n\n')
        file.close()







