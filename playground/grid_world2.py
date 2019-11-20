# maze example
import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from collections import defaultdict
import random
import torch

class Net(torch.nn.Module):
    
    def __init__(self,input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        #self.relu = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.l1(x)
        #x = self.relu(x)
        x = self.l3(x)
        return x

def get_model(learning_rate):
    input_size = 4
    hidden_size = 6
    output_size = 1

    model = Net(input_size,hidden_size,output_size)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer

def update_net(model, loss_fn, optimizer, G_t, state, action):
    optimizer.zero_grad()
    state = torch.from_numpy(state).float()
    action = torch.Tensor(action).float()
    #G_t = torch.Tensor(G_t).float()
    loss = loss_fn(model(torch.cat((state, action), 0)), G_t)
    loss.backward()
    optimizer.step()


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

def printing_policy(policy, value_function):
    print_policy = np.zeros((env._size[0],env._size[1]))
    print_value = np.zeros((env._size[0],env._size[1]))
    for row in range(0, env._size[0]):
        for col in range(0, env._size[1]):
            policy_state_num = ''
            state = tuple([row,col])
            max_index = np.argwhere(policy[state] == np.amax(policy[state]))
            max_index = max_index.flatten().tolist()
            for x in range(0, len(max_index)):
                policy_state_num += str(max_index[x]+1)
            print_policy[state] = int(policy_state_num)
            #print_policy[state] = np.argmax(policy[state])+1
            print_value[state] = value_function[state]
    return print_policy, print_value

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

def sarsa_controlN(num_episodes,e=0.1, gamma=0.9, lr=0.1, n=1):
    sequence_state_counter = defaultdict(lambda: np.zeros(len(env._actions)))
    Q = defaultdict(lambda: np.zeros(len(env._actions)))
    terminal_state_indicators = ["H","N","F","G"]

    for ep in range(0, num_episodes):

        if ep % 1000 == 0:
            print("\rEpisode {}/{}.\n".format(ep, num_episodes), end="")
            sys.stdout.flush()

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
        step_count = 0
        for traj in episode:
            state = traj[0]
            action = traj[1]
            reward = traj[2]
            state_tuple = tuple(state)

            sequence_state_counter[state_tuple][action] += 1.0
            G_t = 0
            gamma_n = min(len(episode), step_count+n)
            for i in range(step_count, gamma_n):
                reward = episode[i][2]
                G_t += reward*(gamma**(i-step_count))
            if((step_count+1) < len(episode)):
                next_state = tuple(episode[step_count+1][0])
                next_action = episode[step_count+1][1]
                #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*Q[next_state][action] - Q[state_tuple][action])
                Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(G_t + (gamma**gamma_n)*Q[next_state][next_action] - Q[state_tuple][action])
            else:
                #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*0.0 - Q[state_tuple][action])
                Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(G_t + gamma*0.0 - Q[state_tuple][action])
            step_count += 1
    return Q

def sarsa_controlNFA(num_episodes,e=0.1, gamma=0.9, lr=0.001, n=1):
    #Q = defaultdict(lambda: np.zeros(len(env._actions)))
    q_model, loss_fn, q_optimizer = get_model(lr)
    #target_model, _, _ = get_model(lr)
    #target_model.load_state_dict(q_model.state_dict())
    terminal_state_indicators = ["H","N","F","G"]

    for ep in range(0, num_episodes):
        # if(ep%10==0):
        #     print("Episode:  " + str(ep))
        #     print_policy = np.zeros((env._size[0],env._size[1]))
        #     for row in range(0,env._size[0]):
        #         for col in range(0,env._size[1]):
        #             state = torch.Tensor([float(row), float(col)])
        #             print_policy[row][col] = my_approx_policy(state,q_model,epsilon)+1
        #     print("0: Terminal State    1: Left    2: Right    3: Up    4: Down")
        #     print(print_policy)
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
            epsilon = 1/(ep+1)
            action = my_approx_policy(state,q_model,epsilon)
            action_index = env._actions.index(action)
            reward = env.next(action_index)
            next_state = env._s
            curr = env.check_state(next_state)
            naction = my_approx_policy(next_state,q_model,epsilon)
            if(curr in terminal_state_indicators):
                done = 1  
            #store experience
            episode.append((state, action, reward, next_state, naction))
            #if done, break
            if done:
                break
            state = next_state
        
        #for replay in range(0,5):
        step_count = 0
        for traj in episode:
            state = traj[0]
            action = traj[1]
            reward = traj[2]

            G_t = 0
            gamma_n = min(len(episode), step_count+n)
            for i in range(step_count, gamma_n):
                reward = episode[i][2]
                G_t += reward*(gamma**(i-step_count))
            if((step_count+1) < len(episode)):
                next_state_torch= torch.from_numpy(traj[3]).float()
                next_action = torch.Tensor(episode[step_count+1][1]).float()

                target_output = q_model(torch.cat((next_state_torch, next_action), 0))
                G_t += (gamma**gamma_n)*target_output.item()
                update_net(q_model, loss_fn, q_optimizer, torch.Tensor([G_t]), state, action)

            else:
                #Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(G_t + gamma*0.0 - Q[state_tuple][action])
                update_net(q_model, loss_fn, q_optimizer, torch.Tensor([G_t]), state, action)
            step_count += 1
        #target_model.load_state_dict(q_model.state_dict())
    return q_model

def my_approx_policy(state, q_model, e):
    #action = random.randint(0,3)
    choice = np.random.choice(2, p=[e, 1-e])
    actions = []
    if choice:
        #greedy
        state = torch.from_numpy(state).float()
        for action in env._actions:
            act = torch.Tensor(action).float()
            actions.append(q_model(torch.cat((state, act), 0)))
        action = actions.index(max(actions))
    else:
        #random
        action = np.random.randint(len(env._actions))
    return  env._actions[int(action)]

def sarsa_run():
    #******************* SARSA ********************************
    #value_function = td_pred(num_episodes=100, gamma=0.99)
    #plot_value_function(value_function, title="Value function")
    value_function = defaultdict(float)
    Q_function = sarsa_controlN(num_episodes=20, gamma=0.9, n=1)
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
    print("0: Terminal State    1: Left    2: Right    3: Up    4: Down")
    print(print_value)

def sarsa_approx_run():
    #******************* SARSA ********************************
    #value_function = td_pred(num_episodes=100, gamma=0.99)
    #plot_value_function(value_function, title="Value function")
    value_function = defaultdict(float)
    Q_model = sarsa_controlNFA(num_episodes=1000, gamma=0.9, n=8)
    print_policy = np.zeros((env._size[0],env._size[1]))
    for row in range(0,env._size[0]):
        for col in range(0,env._size[1]):
            state = np.array([float(row), float(col)])
            print_policy[row][col] = env._actions.index(my_approx_policy(state, Q_model, 0))+1
    print("0: Terminal State    1: Left    2: Right    3: Up    4: Down")
    print(print_policy)
    # for state, actions in Q_function.items():
    #     action_value = np.max(actions)
    #     value_function[state] = action_value

    #     action_state_num = ''
    #     max_index = np.argwhere(actions == np.amax(actions))
    #     max_index = max_index.flatten().tolist()
    #     for x in range(0, len(max_index)):
    #         action_state_num += str(max_index[x]+1)
    #     print_value[state] = int(action_state_num)
    # print("0: Terminal State    1: Left    2: Right    3: Up    4: Down")
    # print(print_value)






env = GridWorld("../grid.txt")
env.print_map()

sarsa_approx_run()

# top-left to (0,0)
def coord_convert(s, sz):
    return [s[1], sz[0]-s[0]-1]


