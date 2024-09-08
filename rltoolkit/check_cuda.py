#!/usr/bin/env python
""" Checks it CUDA is available.

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

import torch

def main():
	if torch.cuda.is_available():
		print("CUDA is available.")
		n_gpus = torch.cuda.device_count()

		print(f"GPUs available: {n_gpus}") # Normally 1

		for i in range(n_gpus):
			print(f"{i}: {torch.cuda.get_device_name(i)}")

		print("You can also try 'nvidia-smi' from the command line.")
	else:
		print("CUDA is not available.")


if __name__ == "__main__":
	main()