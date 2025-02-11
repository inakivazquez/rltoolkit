#!/usr/bin/env python
""" Basic Stable-Baselines3 tester on Gymnasium environments.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Inaki Vazquez"
__email__ = "ivazquez@deusto.es"
__license__ = "GPLv3"

import argparse
import sys
import json
import numpy as np
import importlib

from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from torch import nn

def main():

	parser = argparse.ArgumentParser(description='Trains an environment with an SB3 algorithm and saves the final policy (as well as checkpoints every 50k steps).')
	parser.add_argument('-e', '--env', type=str, required=True, help='environment to test (e.g. CartPole-v1)')
	parser.add_argument('-a', '--algo', type=str, required=True,
						help='algorithm to test from SB3, such as PPO, SAC, DQN... using default hyperparameters')
	parser.add_argument('-n', '--nsteps', type=int, default=100_000, help='number of steps to train')
	parser.add_argument('-y', '--hyperparams', type=str, default=None, help='path to json file with hyperparameters to use in the algorithm instead of the default ones')
	parser.add_argument('--name', type=str, default="model", help='name of this experiment (for logs and policies)')
	parser.add_argument('-v', '--visualize', action="store_true", help='visualize the training with render_mode=\'human\'')
	parser.add_argument('-t', '--tblog', action="store_true", help='generate tensorboard logs in the \"logs\" directory')
	parser.add_argument('-p', '--policy', type=str, default=None, help='policy to load to continue training, it will also read the replay buffer if needed')
	parser.add_argument('-i', '--envpackage', type=str, default=None, help='python package with the environment if not included in Gymnasium')
	parser.add_argument('-m', '--envparams', type=str, default=None, help='path to json file with environment parameters to be passed during creation')
	parser.add_argument('--nenvs', type=int, default=1, help='the number of environments to train in parallel (default=1)')

	args = parser.parse_args()

	str_env = args.env
	str_algo = args.algo
	algo = getattr(sys.modules[__name__], str_algo) # Obtains the classname based on the string
	n_steps = args.nsteps
	tblog_dir = None if args.tblog==False else "./logs"
	experiment_name = args.name
	render_mode = 'human' if args.visualize else None
	policy_file = args.policy
	hyperparams = json.load(open(args.hyperparams)) if args.hyperparams else {}
	envparams = json.load(open(args.envparams)) if args.envparams else {}
	n_envs = args.nenvs
	if args.envpackage:
		importlib.import_module(args.envpackage)

	set_random_seed(42)

	# Create environment
	env = make_vec_env(lambda: gym.make(str_env, render_mode=render_mode, **envparams), n_envs=n_envs)

	print(f"Training for {n_steps} steps with {str_algo}...")

	policy_arch = "MlpPolicy"

	if policy_file:
		# No need to load the hyperparams file as they are already stored in the policy file
		model = algo.load(f"{policy_file}", env=env, tensorboard_log=tblog_dir, verbose=True)
		replay_buffer_file = policy_file.removesuffix("_policy.zip") + "_replay_buffer.pkl"
		if algo in [SAC, TD3, DDPG, DQN]:
			model.load_replay_buffer(replay_buffer_file)
		reset_tblog = False
	else:
		# Instantiate the agent
		model = algo(policy_arch, env=env, tensorboard_log=tblog_dir, verbose=True, **hyperparams)    
		reset_tblog = True

	# Train the agent and display a progress bar
	checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=f"./checkpoints/{experiment_name}_{str_algo}")
	model.learn(total_timesteps=int(n_steps), callback=checkpoint_callback, progress_bar=True, tb_log_name=f"{experiment_name}_{str_algo}", reset_num_timesteps=reset_tblog)

	model.save(f"policies/{experiment_name}_{str_algo}_policy.zip")
	if algo in [SAC, TD3, DDPG, DQN]:
		model.save_replay_buffer(f"./policies/{experiment_name}_{str_algo}_replay_buffer.pkl")

	env.close()

if __name__ == "__main__":
	main()