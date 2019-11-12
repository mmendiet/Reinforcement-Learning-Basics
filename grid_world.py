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
    input_size = 2
    hidden_size = 8
    output_size = 4

    model = Net(input_size,hidden_size,output_size)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return model, loss_fn, optimizer

def update_net(model, loss_fn, optimizer, G_t, state, action):
    optimizer.zero_grad()
    q_values = model(state)
    #q_val = q_values[action]
    loss = loss_fn(q_values[action], G_t)
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

def mc_run():
    #******************* MC ********************************
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

def sarsa_control_online(num_episodes,e=0.1, gamma=0.9, lr=0.1):
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
                Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(reward + gamma*0.0 - Q[state_tuple][action])
                break
            state = next_state
            next_state_tuple = tuple(next_state)
            epsilon = 1/(ep+2)
            next_action = my_policy(next_state_tuple,Q,epsilon)
            #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*Q[next_state][action] - Q[state_tuple][action])
            Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(reward + gamma*Q[next_state_tuple][next_action] - Q[state_tuple][action])

    return Q

def sarsa_controlN_online(num_episodes,e=0.1, gamma=0.9, lr=0.1, n=1):
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
        step_done = -1
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
                #play out remaining steps
                for k in range(step_done+1,len(episode)):
                    initial_idx = max(0,k-(n-1))
                    for i in range(initial_idx, k+1):
                        reward = episode[i][2]
                        G_t += reward*(gamma**(i-initial_idx))
                    if((k+1) < len(episode)):
                        next_state_tuple = tuple(episode[k+1][0])
                        epsilon = 1/(ep+2)
                        next_action = my_policy(next_state_tuple,Q,epsilon)
                        old_state_tuple = tuple(episode[initial_idx][0])
                        #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*Q[next_state][action] - Q[state_tuple][action])
                        Q[old_state_tuple][action] = Q[old_state_tuple][action] + (lr)*(G_t + (gamma**n)*Q[next_state_tuple][next_action] - Q[old_state_tuple][action])
                        step_done += 1
                    else:
                        Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(reward + gamma*0.0 - Q[state_tuple][action])
                break

            if(steps >= (n-1)):
                initial_idx = steps-(n-1)
                G_t = 0.0
                for i in range(initial_idx, steps+1):
                    reward = episode[i][2]
                    G_t += reward*(gamma**(i-initial_idx))
                next_state_tuple = tuple(next_state)
                epsilon = 1/(ep+2)
                next_action = my_policy(next_state_tuple,Q,epsilon)
                old_state_tuple = tuple(episode[initial_idx][0])
                #Q[state_tuple][action] = Q[state_tuple][action] + (1/sequence_state_counter[state_tuple][action])*(reward + gamma*Q[next_state][action] - Q[state_tuple][action])
                Q[old_state_tuple][action] = Q[old_state_tuple][action] + (lr)*(G_t + (gamma**n)*Q[next_state_tuple][next_action] - Q[old_state_tuple][action])
                step_done += 1
            state = next_state
    return Q

def sarsa_control(num_episodes,e=0.1, gamma=0.9, lr=0.1):
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
                Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(reward + gamma*0.0 - Q[state_tuple][action])
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
        if(ep%10==0):
            print("Episode:  " + str(ep))
            print_policy = np.zeros((env._size[0],env._size[1]))
            for row in range(0,env._size[0]):
                for col in range(0,env._size[1]):
                    state = torch.Tensor([float(row), float(col)])
                    print_policy[row][col] = torch.argmax(q_model(state))+1
            print("0: Terminal State    1: Left    2: Right    3: Up    4: Down")
            print(print_policy)
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
            state_torch = torch.from_numpy(state).float()
            action = my_approx_policy(state_torch,q_model,epsilon)

            reward = env.next(action)
            next_state = env._s
            curr = env.check_state(next_state)
            if(curr in terminal_state_indicators):
                done = 1  
            #store experience
            episode.append((state, action, reward, next_state))
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
            state_torch = torch.from_numpy(state).float()

            G_t = 0
            gamma_n = min(len(episode), step_count+n)
            for i in range(step_count, gamma_n):
                reward = episode[i][2]
                G_t += reward*(gamma**(i-step_count))
            if((step_count+1) < len(episode)):
                next_state_torch= torch.from_numpy(traj[3]).float()
                next_action = episode[step_count+1][1]
                #target_output = target_model(next_state_torch)
                target_output = q_model(next_state_torch)
                G_t += (gamma**gamma_n)*target_output[next_action]
                update_net(q_model, loss_fn, q_optimizer, G_t, state_torch, action)
                #G_t += (gamma**gamma_n)*Q[next_state_tuple][next_action]
                #G_t += (gamma**gamma_n)*Qt_net(nest_state)
                #Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(G_t - Q[state_tuple][action])
                #actions_output = Q_net(state_tuple)
                #output = actions_output[action]
                #loss(G_t, output)
                #back_prop and update Q_net

            else:
                #Q[state_tuple][action] = Q[state_tuple][action] + (lr)*(G_t + gamma*0.0 - Q[state_tuple][action])
                update_net(q_model, loss_fn, q_optimizer, torch.Tensor([G_t])[0], state_torch, action)
            step_count += 1
        #target_model.load_state_dict(q_model.state_dict())
    return q_model

def my_approx_policy(state, q_model, e):
    #action = random.randint(0,3)
    choice = np.random.choice(2, p=[e, 1-e])
    if choice:
        #greedy
        action = torch.argmax(q_model(state))
    else:
        #random
        action = np.random.randint(len(env._actions))
    return int(action)

def td_run():
    #******************* TD ********************************
    #value_function = td_pred(num_episodes=100, gamma=0.99)
    #plot_value_function(value_function, title="Value function")
    value_function = defaultdict(float)
    Q_function = td_control(num_episodes=2000, gamma=0.99)
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
    Q_model = sarsa_controlNFA(num_episodes=10000, gamma=0.9, n=8)
    print_policy = np.zeros((env._size[0],env._size[1]))
    for row in range(0,env._size[0]):
        for col in range(0,env._size[1]):
            state = torch.Tensor([float(row), float(col)])
            print_policy[row][col] = torch.argmax(Q_model(state))+1
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






env = GridWorld("grid.txt")
env.print_map()

sarsa_approx_run()

# top-left to (0,0)
def coord_convert(s, sz):
    return [s[1], sz[0]-s[0]-1]


