import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from collections import defaultdict
import random

#This grid_world environment is from the following: https://nbviewer.jupyter.org/url/webpages.uncc.edu/mlee173/teach/itcs6010/notebooks/assign/Assign1.ipynb
class GridWorld:
    """ Grid World environment
            there are four actions (left, right, up, and down) to move an agent
            In a grid, if it reaches a goal, it get 30 points of reward.
            If it falls in a hole or moves out of the grid world, it gets -5.
            Each step costs -1 point. 

        to test GridWorld, run the following sample codes:

            env = GridWorld('grid.txt')

            env.print_map()
            print [2,3], env.check_state([2,3])
            print [0,0], env.check_state([0,0])
            print [3,4], env.check_state([3,4])
            print [10,3], env.check_state([10,3])

            env.init([0,0])
            print env.next(1)  # right
            print env.next(3)  # down
            print env.next(0)  # left
            print env.next(2)  # up
            print env.next(2)  # up

        Parameters
        ==========
        _map        ndarray
                    string array read from a file input
        _size       1d array
                    the size of _map in ndarray
        goal_pos    tuple
                    the index for the goal location
        _actions    list
                    list of actions for 4 actions
        _s          1d array
                    current state
    """
    def __init__(self, fn):
        # read a map from a file
        self._map = self.read_map(fn)
        self._size = np.asarray(self._map.shape)
        self.goal_pos = np.where(self._map == 'G')

        # definition of actions (left, right, up, and down repectively)
        self._actions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self._s = None

    def get_cur_state(self):
        return self._s

    def get_size(self):
        return self._size

    def read_map(self, fn):
        grid = []
        with open(fn) as f:
            for line in f:
               grid.append(list(line.strip()))
        return np.asarray(grid)

    def print_map(self):
        print( self._map )

    def check_state(self, s):
        if isinstance(s, collections.Iterable) and len(s) == 2:
            if s[0] < 0 or s[1] < 0 or\
               s[0] >= self._size[0] or s[1] >= self._size[1]:
               return 'N'
            return self._map[tuple(s)].upper()
        else:
            return 'F'  # wrong input

    def init(self, state=None):
        if state is None:
            s = [0, 0]
        else:
            s = state

        if self.check_state(s) == 'O':
            self._s = np.asarray(state)
        else:
            raise ValueError("Invalid state for init")

    def next(self, a):
        s1 = self._s + self._actions[a]
        # state transition
        curr = self.check_state(s1)
        
        if curr == 'H' or curr == 'N':
            return -5
        elif curr == 'F':
            warnings.warn("invalid state " + str(s1))
            return -5
        elif curr == 'G':
            self._s = s1
            return 30
        else:
            self._s = s1
            return -1
        
    def is_goal(self):
        return self.check_state(self._s) == 'G'
            
    def get_actions(self):
        return self._actions

def policy_evaluation(policy, curr_value_function,num_episodes, gamma=0.9):
    next_value_function = defaultdict(float)
    #terminal_state_indicators = ["H","N","F","G"]
    #states = 
    #for all states
    #get action probabilities
    #have transitional probabiliteis
    for ep in range(0, num_episodes):

        if ep % 1000 == 0:
            print("\rEpisode {}/{}.\n".format(ep, num_episodes), end="")
            sys.stdout.flush()

        for row in range(0, env._size[0]):
            for col in range(0, env._size[1]):
                state = tuple([row,col])

                if(env.check_state([row,col]) == 'O'):
                    for action in range(0,len(env._actions)):#env._actions:
                        env.init([row,col])
                        reward_as = env.next(action)
                        next_state = env._s
                        next_state_tuple = tuple(next_state)
                        tran_prob_a_sTsp = 1
                        prob_aGs = policy[state][action]
                        next_value_function[state] += prob_aGs * (reward_as + gamma*tran_prob_a_sTsp* curr_value_function[next_state_tuple])
                else:
                    #in terminal state
                    curr = env.check_state([row,col])
                    if curr == 'H' or curr == 'N':
                        reward = -5
                    elif curr == 'F':
                        reward = -5
                    elif curr == 'G':
                        reward = 30
                    else:
                        print("Error: Should not have entered this statement")
                    next_value_function[state] += reward
    return next_value_function