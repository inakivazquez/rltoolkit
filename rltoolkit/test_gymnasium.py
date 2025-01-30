#!/usr/bin/env python
""" Basic Gymnasium environments tester.

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

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import argparse
import importlib

def main():

	parser = argparse.ArgumentParser(description='Test Gymnasium environments with random actions.')
	parser.add_argument('-e', '--env', type=str, default="CartPole-v1", help='environment to test (e.g. CartPole-v1)')
	parser.add_argument('-n', '--nsteps', type=int, default=1_000, help='number of steps to execute')
	parser.add_argument('-r', '--recvideo', action="store_true", help='record and store video in a \"video\" directory, instead of using the screen')
	parser.add_argument('-i', '--envpackage', type=str, default=None, help='python package with the environment if not included in Gymnasium')

	args = parser.parse_args()

	str_env = args.env
	n_steps = args.nsteps
	recvideo = args.recvideo
	if args.envpackage:
		importlib.import_module(args.envpackage)

	if not recvideo:
		env = gym.make(str_env, render_mode="human")
	else:
		env = gym.make(str_env, render_mode="rgb_array")
		trigger = lambda n: True
		env = RecordVideo(env, video_folder="./videos", episode_trigger=trigger, disable_logger=True)

	observation, info = env.reset()
	for step_index in range(n_steps):
		action = env.action_space.sample()
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()


	env.close()

if __name__ == "__main__":
	main()