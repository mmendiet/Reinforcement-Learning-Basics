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

def td_pred(num_episodes, gamma=0.9):
    sequence_state_counter = defaultdict(float)
    value_function = defaultdict(float)
    terminal_state_indicators = ["H","N","F","G"]

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
            reward = traj[2]
            state_tuple = tuple(state)

            if state_tuple not in ep_seen_states:
                ep_seen_states.append(state_tuple)
                sequence_state_counter[state_tuple] += 1.0
                if((step_count+1) < len(episode)):
                    next_state = tuple(episode[step_count+1][0])
                    value_function[state_tuple] = value_function[state_tuple] + (1/sequence_state_counter[state_tuple])*(reward + gamma*value_function[next_state] - value_function[state_tuple])
                else:
                    value_function[state_tuple] = value_function[state_tuple] + (1/sequence_state_counter[state_tuple])*(reward + gamma*0.0 - value_function[state_tuple])
            step_count += 1
    return value_function


def td_control(num_episodes,e=0.1, gamma=0.9, lr=0.1):
    sequence_state_counter = defaultdict(lambda: np.zeros(len(env._actions)))
    Q = defaultdict(lambda: np.zeros(len(env._actions)))
    terminal_state_indicators = ["H","N","F","G"]

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
            reward = traj[2]
            state_tuple = tuple(state)

            if state_tuple not in ep_seen_states:
                ep_seen_states.append(state_tuple)
                sequence_state_counter[state_tuple][action] += 1.0
                if((step_count+1) < len(episode)):
                    next_state = tuple(episode[step_count+1][0])
                    next_action = episode[step_count+1][1]
                    #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*Q[next_state][action] - Q[state_tuple][action])
                    Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(reward + gamma*Q[next_state][next_action] - Q[state_tuple][action])
                else:
                    #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*0.0 - Q[state_tuple][action])
                    Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(reward + gamma*0.0 - Q[state_tuple][action])
            step_count += 1
    return Q

env = GridWorld("../grid.txt")
env.print_map()