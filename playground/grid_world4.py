# maze example
import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys
from collections import defaultdict
import random
import math


def plot_value_function(V, title="Value Function", scale_vmin=0):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1])], 2, np.dstack([X, Y]))
    if scale_vmin == 1:
        vmin = np.amin(Z_noace)
        vmax = np.amax(Z_noace)
    else:
        vmax=1.0
        vmin=-1.0
    def plot_surface(X, Y, Z, title, vmin, vmax):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{}".format(title), vmin, vmax)

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

def sarsa_controlN(num_episodes,e=1, gamma=0.9, lr=0.1, n=1):
    Q = defaultdict(lambda: np.zeros(len(env._actions)))
    terminal_state_indicators = ["H","N","F","G"]
    epsilon = e
    final_epsilon = 0.1
    epsilon_decay =  np.exp(np.log(final_epsilon) / 300)

    for ep in range(0, num_episodes):

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
    return Q


def sarsa_run():
    #******************* SARSA ********************************
    #value_function = td_pred(num_episodes=100, gamma=0.99)
    #plot_value_function(value_function, title="Value function")
    value_function = defaultdict(float)
    Q_function = sarsa_controlN(num_episodes=500, gamma=0.9, n=8)
    print_value = np.zeros((env._size[0],env._size[1]))
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
            print_value[state] = 0
        elif(curr in "G"):
            print_value[state] = 9
        else:
            print_value[state] = int(action_state_num)
    print("0: Hole    1: Left    2: Right    3: Up    4: Down    9: Goal")
    print(print_value)


def network_init(lr):
    nHiddens = 40
    nSamples = 1
    nOutputs = 1
    nInputs = 4

    rhoh = rhoo = lr

    rh = rhoh / (nSamples*nOutputs)
    ro = rhoo / (nSamples*nOutputs)

    # Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
    V = 0.1*2*(np.random.uniform(size=(nInputs+1,nHiddens))-0.5)
    W = 0.1*2*(np.random.uniform(size=(1+nHiddens,nOutputs))-0.5)

    return rh, ro, V, W

def forward(X,V,W):
    #X = stdX.standardize(X)
    # Forward pass on training data
    X = (X-np.mean(X))/np.std(X)
    X1 = addOnes(X)
    Z = np.tanh(X1 @ V)
    Z1 = addOnes(Z)
    Y = Z1 @ W
    return Y, Z

def backward(error, Z, X,rh,ro,W):
    ### make sure the array shapes
    X = as_array(X)
    Z = as_array(Z)
    E = as_array(error)
    
    Z1 = addOnes(Z)
    X1 = addOnes(X)

    # Backward pass - the backpropagation and weight update steps
    dV = rh * X1.T @ ( ( E @ W[1:,:].T) * (1-Z**2))
    dW = ro * Z1.T @ E
    return dV, dW

def addOnes(A):
    return np.insert(A, 0, 1, axis=len(np.array(A).shape)-1)

def as_array(A):
    A = np.array(A)
    if len(A.shape) == 1:
        return A.reshape((1, -1))
    return A

def network_update(s,a,r1,s1,a1,gamma,V,W,rh,ro):
    a1buf = np.zeros((a1.shape[0],2))
    for idx in range(0,a1.shape[0]):
        a1buf[idx] = env._actions[a1[idx][0]]
    Q1, _ = forward(np.hstack((s1, a1buf)),V,W)  # output of neural network is Q for next state

    abuf = np.zeros((a.shape[0],2))
    for idx in range(0,a.shape[0]):
        abuf[idx] = env._actions[a[idx][0]]
    Q, Z = forward(np.hstack((s, abuf)),V,W)  # output of neural network is Q for next state

    error = r1 + gamma * Q1 - Q  # use action value as index by adding one
    dV, dW = backward(error, Z, np.hstack((s, abuf)),rh,ro,W)
    V += dV
    W += dW

def epsilon_greedy(e, s, V, W):
    if np.random.rand() < e:
        return np.random.randint(len(env._actions))
    else:
        all_actions = env._actions #grab all actions in action space
        Q, _ = forward(np.hstack((np.tile(s, (4,1)), all_actions)),V,W)
        max_as = np.where(Q == np.max(Q))[0] # index to action value
        return np.random.choice(max_as)

def sarsa_controlNFA(num_episodes,e=1, gamma=0.9, lr=0.1, n=1, batch_size=8):
    terminal_state_indicators = ["H","N","F","G"]
    epsilon = e
    final_epsilon = 0.1
    epsilon_decay =  np.exp(np.log(final_epsilon) / num_episodes)

    rh, ro, V, W = network_init(lr)
    all_Gt = []
    for ep in range(0, num_episodes):

        if(ep%100==0):
            print("Episode:  " + str(ep))
            print_policy(V,W)

        episode = []
        start_row = random.randint(0,env._size[0])
        start_col = random.randint(0,env._size[1])
        if(env.check_state([start_row,start_col]) == 'O'):
            env.init([start_row,start_col])
        else:
            env.init([0,0])
        done = 0
        reward_sum = 0
        for steps in range(0,1000):
            state = env.get_cur_state()
            action = epsilon_greedy(epsilon, state, V, W)

            reward = env.next(action)
            #gather reward_sum for plotting
            reward_sum += reward
            next_state = env._s
            next_action = epsilon_greedy(epsilon, next_state, V, W)
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

        #Early stopping
        last = max(0,ep-1)
        if(all_Gt[last] > reward_sum) and (ep>int(num_episodes*0.8)) and (reward_sum > 0):
            break

        step_count = 0
        G_t_lst = []
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
            G_t_lst.append(G_t)
            step_count += 1
        #prepare batch
        ep_arr = np.asarray(episode)
        b_size = batch_size
        for batch in range(math.ceil(len(ep_arr)/b_size)):
            place = batch*b_size
            limit = min(place+b_size, len(ep_arr))
            size = limit-place
            state_arr = np.concatenate(ep_arr[place:limit,0], axis=0).reshape(size,2).astype(np.int64)
            action_arr = ep_arr[place:limit,1].reshape(size,1).astype(np.int64)
            next_state_arr = np.concatenate(ep_arr[place:limit,3], axis=0).reshape(size,2).astype(np.int64)
            next_action_arr = ep_arr[place:limit,4].reshape(size,1).astype(np.int64)
            G_t_arr = np.asarray(G_t_lst[place:limit]).reshape(size,1).astype(np.int64)
            #run network update
            network_update(state_arr,action_arr,G_t_arr,next_state_arr,next_action_arr,(gamma**n),V,W,rh,ro)
        epsilon *= epsilon_decay
    return V,W, all_Gt

def print_policy(V,W):
    print_value = np.zeros((env._size[0],env._size[1]))
    bad_state_indicators = ["H","N","F"]

    for row in range(0,env._size[0]):
        for col in range(0,env._size[1]):
            state = np.asarray([row,col])

            action_state_num = epsilon_greedy(0, state, V, W) + 1
            curr = env.check_state(state)
            if(curr in bad_state_indicators):
                print_value[tuple(state)] = 0
            elif(curr in "G"):
                print_value[tuple(state)] = 9
            else:
                print_value[tuple(state)] = int(action_state_num)
    print("0: Hole    1: Left    2: Right    3: Up    4: Down    9: Goal")
    print(print_value)

def sarsa_approx_run():
    #******************* SARSA ********************************
    #value_function = td_pred(num_episodes=100, gamma=0.99)
    #plot_value_function(value_function, title="Value function")
    V, W , all_rewards = sarsa_controlNFA(num_episodes=500, gamma=0.9, n=1, lr=0.0002, batch_size=1000)
    print_policy(V,W)
    plt.plot(all_rewards)
    plt.show()

np.set_printoptions(suppress=True)
env = GridWorld("grid.txt")
env.print_map()

sarsa_approx_run()