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

def policy_improvement(curr_policy, value_function, gamma=0.9):
    q_values = defaultdict(dict)
    #for all states
    #find the q values for a state
    #then change policy to act greedily
    for row in range(0, env._size[0]):
        for col in range(0, env._size[1]):
            state = tuple([row,col])

            if(env.check_state([row,col]) == 'O'):
                q_value = np.zeros(len(env._actions))
                for action in range(0,len(env._actions)):#env._actions:
                    env.init([row,col])
                    reward_as = env.next(action)
                    next_state = env._s
                    next_state_tuple = tuple(next_state)
                    tran_prob_a_sTsp = 1

                    #state_action = tuple([state,action])
                    q_value[action] = reward_as + gamma*tran_prob_a_sTsp* value_function[next_state_tuple]
                q_values[state] = q_value
            else:
                #in terminal state
                curr = env.check_state([row,col])
                reward = 0
                if curr == 'H' or curr == 'N':
                    reward = -5
                elif curr == 'F':
                    reward = -5
                elif curr == 'G':
                    reward = 30
                else:
                    print("Error: Should not have entered this statement")
                rewards = np.full(len(env._actions), reward)
                q_values[state] = rewards
            #find best action and form array

            max_index = np.argwhere(q_values[state] == np.amax(q_values[state]))
            max_index = max_index.flatten().tolist()
            new_prob = np.zeros(len(env._actions))
            for idx in range(0, len(max_index)):
                new_prob[max_index[idx]] = 1/len(max_index)
            curr_policy[state] = new_prob
    return curr_policy
def dp_policy_creation():
    policy = defaultdict(dict)
    default_state_prob = np.full(len(env._actions), (1/len(env._actions)))
    for row in range(0, env._size[0]):
        for col in range(0, env._size[1]):
            state = tuple([row,col])
            policy[state] = default_state_prob
    return policy

def dp_run():
    #******************* DP ********************************
    policy = dp_policy_creation()
    np.set_printoptions(suppress=True)
    for i in range(0,100):
        value_function = defaultdict(float)
        print(i)
        value_function = policy_evaluation(policy, value_function, num_episodes=50)
        mock_policy = policy.copy()
        new_policy = policy_improvement(mock_policy, value_function)
        
        converged=1
        for row in range(0, env._size[0]):
            for col in range(0, env._size[1]):
                state = tuple([row,col])
                if(np.all(policy[state] != new_policy[state])):
                    converged = 0
        if(converged==1):
            print("Converged")
            break
        # print_policy, print_value = printing_policy(policy, value_function)
        # print_policy2, print_value = printing_policy(new_policy, value_function)
        # print(print_policy)
        # print(print_policy2)
        policy = new_policy
    
    plot_value_function(value_function, title="Value function", scale_vmin=1)
    print_policy, print_value = printing_policy(policy, value_function)


    print("1: Left    2: Right    3: Up    4: Down")
    print(print_policy)
    #print(print_value)

env = GridWorld("../grid.txt")
env.print_map()