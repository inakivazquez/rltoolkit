# rltoolkit
Basic general purpose tools for experimenting with Reinforcement Learning and Gymnasium environments.

To use the RL Toolkit, you can run the following scripts, directly from the command-line or using the source python file:

1. `check-cuda` (`check_cuda.py`): This script checks if CUDA is properly installed and available on your system. To run the script, use the following command:
    ```
    check-cuda
    ```

2. `test-gymnasium` (`test_gymnasium.py`): This script tests a Gymnasium environment passed as argument, selecting random actions from the action space. To run the script and check available options, use the following command:
    ```
    test-gymnasium --help
    ```
    Example:
    ```
    test-gymnasium -e CartPole-v1 -n 500
    ```

3. `test_sb3`: This script tests the integration of Gymnasium environments with the Stable Baselines3 library. To run the script and check available options, use the following command:
    ```
    test-sb3 --help
    ```
    Example:
    ```
    test-sb3 -e CartPole-v1 --algo PPO -n 10000
    ```

Make sure you have the necessary dependencies installed before running these scripts.
