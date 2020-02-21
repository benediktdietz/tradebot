import sys, os
import numpy as np
import pandas as pd
import pprint, csv, tqdm
from matplotlib import pyplot as plt
import yfinance as yf
import gym
import tensorflow as tf
import torch
from env.StockTradingEnv import StockTradingEnv
from render.StockTradingGraph import StockTradingGraph
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import chainer, chainerrl
from chainer import functions as F
from chainer import links as L
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainer import serializers

from torch.utils.tensorboard import SummaryWriter
os.system('clear')

outdir = './output/3/'

print('\nwriting to ', outdir, '\n')

model_outdir = outdir + 'model/'
tb_outdir = outdir + 'tensorboard/'
path_to_symbol_csv = './sp500.csv'


class rl_stock_trader():

	def __init__(self, path_to_symbol_csv, request_symbols = 8, tb_outdir = tb_outdir):

		self.writer = SummaryWriter(tb_outdir)

		self.request_symbols = request_symbols
		self.monitor_freq = 100

		self.start_budget = 10000.

		index_df = pd.read_csv(path_to_symbol_csv)

		# symbol_vec = list(index_df.values[:self.request_symbols,0])

		symbol_vec = list(index_df.values[
			np.random.randint(
				0,
				index_df.values.shape[0], 
				self.request_symbols), 
			0])


		self.dataframe, self.num_symbols = self.get_data(symbol_vec)


		# env = DummyVecEnv([lambda: StockTradingEnv(dataframe)])
		self.env = StockTradingEnv(self.dataframe, self.num_symbols)



		self.tb_action_type = np.zeros(3)
		self.tb_action_symbol = np.zeros(self.num_symbols)
		self.tb_action_vec = []
		self.tb_action_amount = []

		self.tb_balance = np.zeros(4)
		self.tb_net_worth = np.zeros(4)

		self.balance_dummy = []
		self.net_worth_dummy = []
		self.tb_reward = 0.

		self.tb_cache_reward_vec = []
		self.tb_cache_rollout_vec = []

		self.tb_cache_final_net = []
		self.tb_cache_final_balance = []

		self.tb_chache_balance = np.zeros(4)
		self.tb_chache_net_worth = np.zeros(4)

	def get_data(self, symbols, start=None, end=None, period='5y', interval='1d'):
		'''	valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
			fetch data by interval (including intraday if period < 60 days)
			valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
			group by ticker (to access via data['SPY']) (optional, default is 'column')
			adjust all OHLC automatically
			download pre/post regular market hours data
			use threads for mass downloading? (True/False/Integer)
			proxy URL scheme use use when downloading? '''

		df_keys = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']


		if start == None or end == None:

			print('\nload S&P 500 data for period: ', period, ' and interval: ', interval, '\n')

			data_array = yf.download(
				tickers = symbols,
				period = period,
				interval = interval,
				group_by = 'column',
				auto_adjust = True,
				prepost = False,
				threads = True,
				proxy = None
				)

		else:

			print('\nload S&P 500 data since: ', start, '/ end: ', end, ' and interval: ', interval, '\n')

			data_array = yf.download(
				tickers = symbols,
				start = start, 
				end = end,
				interval = interval,
				group_by = 'column',
				auto_adjust = True,
				prepost = False,
				threads = True,
				proxy = None
				)

		called_symbols = list(data_array['Volume'].keys())
		try:
			failed_symbols = list(data_array['Adj Close'].keys())
		except KeyError:
			failed_symbols = []
			pass

		loaded_symbols = []

		for i in range(len(called_symbols)):
			if called_symbols[i] not in failed_symbols:
				loaded_symbols.append(called_symbols[i])

		for i in range(len(failed_symbols)):
			for j in range(len(df_keys)):
				data_array = data_array.drop(columns=[(str(df_keys[j]), str(failed_symbols[i]))])


		data_array.insert(0, 'i', np.arange(data_array.shape[0]))


		data_index_axis = data_array.index.values
		data_array = data_array.drop(index = [data_index_axis[0], data_index_axis[-1]])




		dfkeys = ['Open', 'Close', 'High', 'Low', 'Volume']

		for dfkey in range(len(dfkeys)):

			data_array[dfkeys[dfkey]].fillna(method ='pad') 
			data_array[dfkeys[dfkey]].fillna(0.)  
			data_array[dfkeys[dfkey]].replace(to_replace = np.nan, value = 0.)
			data_array[dfkeys[dfkey]].replace(to_replace = 'NaN', value = 0.)



		print(
			'\n------------------------------------\
			\nsuccesfully loaded stock data\nnumber of loaded data points: ', data_array.shape[0], \
			'\nnumber of loaded symbols: ', len(loaded_symbols), '/', len(called_symbols), \
			'\n------------------------------------\n\n', \
			'\ndataframe:\n', data_array, \
			'\n------------------------------------\n\n')

		return data_array, len(loaded_symbols)

	def monitor_training(self, tb_writer, t, i, done, action, monitor_data):

		'''
		after each episode save:
			action_type [3 x 1]  v
			action_amount [1 x 1] (avg /t)  v
			action_symbol [num_symbols x 1]  v
			balance [4x1] (low, avg, high, final)  v
			net_worth [4x1] (low, avg, high, final)  v

		'''


		if t == 0: 

			self.balance_dummy = []
			self.net_worth_dummy = []
			self.tb_reward = 0.

			if i == 0:


				self.tb_balance = np.zeros(4)
				self.tb_net_worth = np.zeros(4)

				self.tb_action_amount = []
				self.tb_action_symbol_vec = []

				self.tb_action_vec = []

				self.tb_cache_reward_vec = []
				self.tb_cache_rollout_vec = []

				self.tb_cache_final_net = np.zeros(4)
				self.tb_cache_final_balance = np.zeros(4)




		self.tb_action_symbol_vec.append(monitor_data['action_sym'])
			
		self.tb_action_amount.append(monitor_data['action_amount'])

		self.tb_action_vec.append(monitor_data['action_type'])





		self.tb_reward += monitor_data['reward']

		self.balance_dummy.append(monitor_data['balance'])
		self.net_worth_dummy.append(monitor_data['net_worth'])


		if done:

			self.tb_cache_reward_vec.append(self.tb_reward)

			self.tb_balance[0] = np.amin(self.balance_dummy)
			self.tb_balance[1] = np.mean(self.balance_dummy)
			self.tb_balance[2] = np.amax(self.balance_dummy)
			self.tb_balance[3] = self.balance_dummy[-1]

			self.tb_net_worth[0] = np.amin(self.net_worth_dummy)
			self.tb_net_worth[1] = np.mean(self.net_worth_dummy)
			self.tb_net_worth[2] = np.amax(self.net_worth_dummy)
			self.tb_net_worth[3] = self.net_worth_dummy[-1]

			self.tb_cache_rollout_vec.append(t)


			if np.ndim(self.tb_cache_final_balance) == 1:
				self.tb_cache_final_balance = np.reshape(self.tb_balance, [1,-1])
				self.tb_cache_final_net = np.reshape(self.tb_net_worth, [1,-1])
			else:
				self.tb_cache_final_balance = np.concatenate((self.tb_cache_final_balance, np.reshape(self.tb_balance, [1,-1])), axis=0)
				self.tb_cache_final_net = np.concatenate((self.tb_cache_final_net, np.reshape(self.tb_net_worth, [1,-1])), axis=0)


			if i % self.monitor_freq == 0 and i != 0:

				tb_writer.add_scalar('training/reward', np.mean(self.tb_cache_reward_vec), i)
				tb_writer.add_scalar('training/rollout', np.mean(self.tb_cache_rollout_vec), i)

				tb_writer.add_scalar('balance/low', np.mean(self.tb_cache_final_balance[:,0]), i)
				tb_writer.add_scalar('balance/avg', np.mean(self.tb_cache_final_balance[:,1]), i)
				tb_writer.add_scalar('balance/high', np.mean(self.tb_cache_final_balance[:,2]), i)
				tb_writer.add_scalar('balance/final', np.mean(self.tb_cache_final_balance[:,3]), i)

				tb_writer.add_scalar('net_worth/low', np.mean(self.tb_cache_final_net[:,0]), i)
				tb_writer.add_scalar('net_worth/avg', np.mean(self.tb_cache_final_net[:,1]), i)
				tb_writer.add_scalar('net_worth/high', np.mean(self.tb_cache_final_net[:,2]), i)
				tb_writer.add_scalar('net_worth/final', np.mean(self.tb_cache_final_net[:,3]), i)
				tb_writer.add_scalar('net_worth/profit', np.mean(self.tb_cache_final_net[:,3] - self.start_budget), i)



				tb_writer.add_histogram('training_stats/reward', np.asarray(self.tb_cache_reward_vec), i)
				tb_writer.add_histogram('training_stats/rollout', np.asarray(self.tb_cache_rollout_vec), i)

				tb_writer.add_histogram('performance_stats/final_balance', np.asarray(self.tb_cache_final_balance[:,-1]), i)
				tb_writer.add_histogram('performance_stats/final_net_worth', np.asarray(self.tb_cache_final_net[:,-1]), i)
				tb_writer.add_histogram('performance_stats/profit', np.asarray(self.tb_cache_final_net[:,-1] - self.start_budget), i)

				tb_writer.add_histogram('action/type', np.asarray(self.tb_action_vec), i)
				tb_writer.add_histogram('action/symbol', np.asarray(self.tb_action_symbol_vec), i)
				tb_writer.add_histogram('action/action_amount', np.asarray(self.tb_action_amount), i)



				self.tb_cache_reward_vec = []
				self.tb_cache_rollout_vec = []

				self.tb_cache_final_net = np.zeros(4)
				self.tb_cache_final_balance = np.zeros(4)

				self.tb_action_vec = []

				self.tb_action_symbol_vec = []

				self.tb_action_amount = []

				self.tb_balance = np.zeros(4)
				self.tb_net_worth = np.zeros(4)

	def rl_agent(self, env):

		self.policy = chainer.Sequential(
			L.Linear(None, 256),
			F.tanh,
			L.Linear(None, 128),
			F.tanh,
			# L.Linear(None, env.action_space.low.size, initialW=winit_last),
			L.Linear(None, env.action_space.low.size),
			# F.sigmoid,
			chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
				action_size=env.action_space.low.size,
				var_type='diagonal',
				var_func=lambda x: F.exp(2 * x),  # Parameterize log std
				# var_param_init=0,  # log std = 0 => std = 1
				))



		self.vf = chainer.Sequential(
			L.Linear(None, 256),
			F.tanh,
			L.Linear(None, 128),
			F.tanh,
			L.Linear(None, 1),
		)

		# Combine a policy and a value function into a single model
		self.model = chainerrl.links.Branched(self.policy, self.vf)

		self.opt = chainer.optimizers.Adam(alpha=3e-4, eps=1e-5)
		self.opt.setup(self.model)


		self.agent = PPO(self.model, 
					self.opt,
					# obs_normalizer=obs_normalizer,
					gpu=-1,
					update_interval=512,
					minibatch_size=8,
					clip_eps_vf=None, 
					entropy_coef=0.001,
					# standardize_advantages=args.standardize_advantages,
					)

		return self.agent

	def train(self):


		print('\nstart training loop\n')

		def check_types(input, inputname):
			if np.isnan(input).any(): print('----> ', inputname, ' array contains NaN\n', np.isnan(input).shape, '\n')
			if np.isinf(input).any(): print('----> ', inputname, ' array contains inf\n', np.isinf(input).shape, '\n')



		self.agent = self.rl_agent(self.env)


		n_episodes = 1000000
		max_episode_len = 1000


		for i in range(0, n_episodes + 1):


			obs = self.env.reset()

			reward = 0
			done = False
			R = 0  # return (sum of rewards)
			t = 0  # time step


			while not done and t < max_episode_len:

				# Uncomment to watch the behaviour
				# self.env.render()
				action = self.agent.act_and_train(obs, reward)
				check_types(action, 'action')

				obs, reward, done, _, monitor_data = self.env.step(action)
				check_types(obs, 'obs')
				check_types(reward, 'reward')

				self.monitor_training(self.writer, t, i, done, action, monitor_data)

				R += reward
				t += 1

				if done: print(' training at episode ' + str(i), end='\r')


			if i % 100 == 0 and i > 0:

				self.agent.save(model_outdir)

				serializers.save_npz(model_outdir + 'model.npz', self.model)

			
			# if i % 1000 == 0:
			#     print('\nepisode:', i, ' | episode length: ', t, '\nreward:', R,
			#           '\nstatistics:', self.agent.get_statistics(), '\n')

		self.agent.stop_episode_and_train(obs, reward, done)
		print('Finished.')


rl_stock_trader(path_to_symbol_csv).train()



















