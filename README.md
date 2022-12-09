# A2C and REINFORCE

The main file can be used to generate the results. 

Dependencies - Would need PyTorch, Gym, and PyBullet using Bullet gym environments

Usage- From the directory A2C, run python main.py with the following arguments
--env
--algo (REINFORCE Or A2C)
--mem_steps
--total_steps

Note- Using same random seeds does not help since we ae dealing with stochastic algorithms. Sometimes, A2C does not give good performance, kindly run 3-4 times for evaluating the learning curves, it takes less than 1 minute for one run on GPU (atleast my system)


Test cases for reproducibility - 
*Please run from terminal the following commands
A2C
1) python main.py --env Pendulum-v1 --algo A2C --mem_steps 32 --learning_steps 5000

2) python main.py --env CartPoleContinuousBulletEnv-v0 --algo A2C --mem_steps 32 --learning_steps 5000

3) python main.py --env InvertedDoublePendulumBulletEnv-v0 --algo A2C --mem_steps 32 --learning_steps 2500

4) python main.py --env MountainCarContinuous-v0 --algo A2C --mem_steps 32 --learning_steps 20000

5) python main.py --env Walker2DBulletEnv-v0 --algo A2C --mem_steps 32 --learning_steps 12500

REINFORCE
1) python main.py --env Pendulum-v1 --algo REINFORCE --mem_steps 64 --learning_steps 10000

2) python main.py --env CartPoleContinuousBulletEnv-v0 --algo REINFORCE --mem_steps 64 --learning_steps 5000

3) python main.py --env InvertedDoublePendulumBulletEnv-v0 --algo REINFORCE --mem_steps 64 --learning_steps 2500

4) python main.py --env MountainCarContinuous-v0 --algo REINFORCE --mem_steps 64 --learning_steps 15000

5) python main.py --env Walker2DBulletEnv-v0 --algo REINFORCE --mem_steps 64 --learning_steps 7500



Evaluation of Learnt Agent
For Pendulum-v1, please follow the following steps, run these commands from different terminals

1) First run the following command and wait until learning is finished, it will show a plot.

python main.py --env Pendulum-v1 --algo A2C --mem_steps 32 --learning_steps 5000

2) Open a new terminal and run the following to give rewards for 250 episodes of the learned policy

python main.py --env Pendulum-v1 --algo A2C --mem_steps 32 --learning_steps 5000 --learn 0 --random 0
#learned policy evaluation

3) Open a new terminal and run the following to give rewards for 250 episodes of the random policy

python main.py --env Pendulum-v1 --algo A2C --mem_steps 32 --learning_steps 5000 --learn 0 --random 1
#random policy evaluation

