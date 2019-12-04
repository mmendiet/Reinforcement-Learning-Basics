
import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from collections import defaultdict
import random
import math
import time
from GridWorld import *
from utils import *



def sarsa_controlN(num_episodes,e=1, gamma=0.9, lr=0.1, n=1):
    Q = defaultdict(lambda: np.zeros(len(env._actions)))
    terminal_state_indicators = ["H","N","F","G"]
    epsilon = e
    final_epsilon = 0.1
    epsilon_decay =  np.exp(np.log(final_epsilon) / 300)

    all_Gt = []
    for ep in range(0, num_episodes):
        reward_sum = 0
        episode = []
        start_row = random.randint(0,env._size[0])
        start_col = random.randint(0,env._size[1])
        if(env.check_state([start_row,start_col]) == 'O'):
            env.init([start_row,start_col])
        else:
            env.init([0,0])
        done = 0
        for steps in range(0,1000):
            state = env.get_cur_state()
            action = my_policy(tuple(state),Q,epsilon)

            reward = env.next(action)
            #gather reward_sum for plotting
            reward_sum += reward
            
            next_state = env._s
            next_action = my_policy(tuple(next_state),Q,epsilon)
            curr = env.check_state(next_state)
            if(curr in terminal_state_indicators):
                done = 1  
            #store experience
            episode.append((state, action, reward, next_state, next_action))
            #if done, break
            if done:
                break
            state = next_state
        all_Gt.append(reward_sum)
        step_count = 0
        for traj in episode:
            state = traj[0]
            action = traj[1]
            reward = traj[2]
            next_state = traj[3]
            next_action = traj[4]

            G_t = 0
            n_step = min(len(episode), step_count+n)
            for i in range(step_count, n_step):
                reward = episode[i][2]
                G_t += reward*(gamma**(i-step_count))
            gamma_n = n_step-step_count
            Q[tuple(state)][action] = Q[tuple(state)][action] + (lr)*(G_t + (gamma**gamma_n)*Q[tuple(next_state)][next_action] - Q[tuple(state)][action])
            step_count += 1
        epsilon *= epsilon_decay
    return Q, all_Gt

def my_policy(state, Q, e):
    #action = random.randint(0,3)
    choice = np.random.choice(2, p=[e, 1-e])
    if choice:
        #greedy
        action = np.argmax(Q[state])
    else:
        #random
        action = np.random.randint(len(env._actions))
    return action

def sarsa_run(steps):
    #******************* SARSA ********************************
    value_function = defaultdict(float)
    start = time.time()
    Q_function , all_reward = sarsa_controlN(num_episodes=500, gamma=0.9, n=steps)
    end = time.time()
    print_policy = np.zeros((env._size[0],env._size[1]))
    bad_state_indicators = ["H","N","F"]

    for state, actions in Q_function.items():
        action_value = np.max(actions)
        value_function[state] = action_value

        action_state_num = ''
        max_index = np.argwhere(actions == np.amax(actions))
        max_index = max_index.flatten().tolist()
        for x in range(0, len(max_index)):
            action_state_num += str(max_index[x]+1)
        curr = env.check_state(state)
        if(curr in bad_state_indicators):
            print_policy[state] = 0
        elif(curr in "G"):
            print_policy[state] = 9
        else:
            print_policy[state] = int(action_state_num)
    time_passed = (end-start)
    return time_passed, print_policy, all_reward

    
np.set_printoptions(suppress=True)
env = GridWorld("../grid.txt")
env.print_map()

time1, print_policy, all_reward = sarsa_run(1)
print("\n\nSARSA: 1 Step      Time: " + str(time1) + " seconds")
print("0: Hole    1: Left    2: Right    3: Up    4: Down    9: Goal")
print(print_policy)
fig = plt.figure()
plt.plot(all_reward)
fig.suptitle('SARSA 1 Step', fontsize=20)
print(all_reward[-1])

time4, print_policy, all_reward = sarsa_run(4)
print("\n\nSARSA: 4 Step      Time: " + str(time4) + " seconds")
print("0: Hole    1: Left    2: Right    3: Up    4: Down    9: Goal")
print(print_policy)
fig = plt.figure()
plt.plot(all_reward)
fig.suptitle('SARSA 4 Step', fontsize=20)
print(all_reward[-1])

time8, print_policy, all_reward = sarsa_run(8)
print("\n\nSARSA: 8 Step      Time: " + str(time8) + " seconds")
print("0: Hole    1: Left    2: Right    3: Up    4: Down    9: Goal")
print(print_policy)
fig = plt.figure()
plt.plot(all_reward)
fig.suptitle('SARSA 8 Step', fontsize=20)
print(all_reward[-1])