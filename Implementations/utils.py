import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from collections import defaultdict
import random

def plot_value_function(V, title="Value Function", scale_vmin=0):

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

def printing_policy(policy, value_function):
    #TODO include env as argument for use as util
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