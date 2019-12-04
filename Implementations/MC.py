import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from collections import defaultdict
import random
from GridWorld import *
from utils import *

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

def mc_pred(num_episodes, gamma=0.9):
    sequence_state_counter = defaultdict(float)
    value_function = defaultdict(float)
    terminal_state_indicators = ["H","N","F","G"]
    #alpha = 0.1

    for ep in range(0, num_episodes):

        if ep % 1000 == 0:
            print("\rEpisode {}/{}.\n".format(ep, num_episodes), end="")
            sys.stdout.flush()

        #ep_state_counter = defaultdict(float)
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
            action = my_policy(state)

            reward = env.next(action)
            next_state = env._s
            curr = env.check_state(next_state)
            if(curr in terminal_state_indicators):
                done = 1  
            #store experience
            episode.append((state, action, reward))
            #if done, break
            if done:
                break
            state = next_state

        ep_seen_states = []
        step_count = 0
        for traj in episode:
            state = traj[0]
            state_tuple = tuple(state)

            if state_tuple not in ep_seen_states:
                ep_seen_states.append(state_tuple)
                sequence_state_counter[state_tuple] += 1.0
                G_t = 0
                for i in range(step_count, len(episode)):
                    reward = episode[i][2]
                    G_t += reward*(gamma**(i-step_count))
                value_function[state_tuple] = value_function[state_tuple] + (1/sequence_state_counter[state_tuple])*(G_t - value_function[state_tuple])
                #value_function[state_tuple] = value_function[state_tuple] + alpha*(G_t - value_function[state_tuple])
            step_count += 1
    return value_function

def mc_control(num_episodes,e=0.1, gamma=0.9, lr=0.1):
    sequence_state_counter = defaultdict( lambda: np.zeros(len(env._actions)))
    Q = defaultdict(lambda: np.zeros(len(env._actions)))
    terminal_state_indicators = ["H","N","F","G"]
    #alpha = 0.1

    for ep in range(0, num_episodes):
        if ep % 1000 == 0:
            print("\rEpisode {}/{}.\n".format(ep, num_episodes), end="")
            sys.stdout.flush()

        #ep_state_counter = defaultdict(float)
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
            state_tuple = tuple(state)

            epsilon = 1/(ep+1)
            action = my_policy(state_tuple,Q,epsilon)

            reward = env.next(action)
            next_state = env._s
            curr = env.check_state(next_state)
            if(curr in terminal_state_indicators):
                done = 1  
            #store experience
            episode.append((state, action, reward))
            #if done, break
            if done:
                break
            state = next_state

        ep_seen_states = []
        step_count = 0
        for traj in episode:
            state = traj[0]
            action = traj[1]
            state_tuple = tuple(state)

            if state_tuple not in ep_seen_states:
                ep_seen_states.append(state_tuple)
                sequence_state_counter[state_tuple][action] += 1.0
                G_t = 0
                for i in range(step_count, len(episode)):
                    reward = episode[i][2]
                    G_t += reward*(gamma**(i-step_count))
                #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(G_t - Q[state_tuple][action])
                Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(G_t - Q[state_tuple][action])
            step_count += 1
    return Q

def mc_run():
    #******************* MC *******************************
    #value_function = mc_pred(num_episodes=1, gamma=0.99)
    value_function = defaultdict(float)
    Q_function = mc_control(num_episodes=2000, gamma=0.99)
    print_value = np.zeros((env._size[0],env._size[1]))
    for state, actions in Q_function.items():
        action_value = np.max(actions)
        value_function[state] = action_value

        action_state_num = ''
        max_index = np.argwhere(actions == np.amax(actions))
        max_index = max_index.flatten().tolist()
        for x in range(0, len(max_index)):
            action_state_num += str(max_index[x]+1)
        print_value[state] = int(action_state_num)
    print("1: Left    2: Right    3: Up    4: Down")
    print(print_value)
    #plot_value_function(value_function, title="Value function")

env = GridWorld("../grid.txt")
env.print_map()