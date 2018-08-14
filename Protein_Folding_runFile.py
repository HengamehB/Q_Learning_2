# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from gym import spaces
from gym_lattice.envs import Lattice2DEnv
import numpy as np
import random

np.random.seed(42)

seq = 'HPhP' # Our input sequence
action_space = spaces.Discrete(4) # Choose among [0, 1, 2 ,3]
N_EPISODES = 100
MAX_EPISODE_STEPS = len(seq)
env = Lattice2DEnv(seq)

MIN_ALPHA = 0.0001
alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)

MIN_EPSILON = 0.05
epsilons = np.linspace(0.9, MIN_EPSILON, N_EPISODES)
gamma = 0.95

q_table = dict()

################# Functions #################

def choose_action(state,eps):
    if random.uniform(0, 1) < eps:
        return action_space.sample() 
    else:
        return np.argmax(q(state))
    
def q(state, action=None):
    
    state = repr(state)
    
    if state not in q_table:
        q_table[state] = np.zeros(4)
        
    if action is None:
        return q_table[state]
    
    return q_table[state][action]

#############################################

for i_episodes in range(N_EPISODES):
    
#    print (i_episodes)
    
    env.reset()
    Total_reward = 0
    alpha = alphas[i_episodes]
    i_step = 0
    eps = epsilons[i_episodes]
#    eps = MIN_EPSILON
    
    while True:
        
        # Random agent samples from action space ---> No longer Valid 
        action = choose_action(env.state,eps)
        old_state, new_state, reward, done, info, grid = env.step(action)
        
        if old_state != new_state:
            i_step += 1
            
        Total_reward += reward
        
        q(old_state)[action] = q(old_state, action) + \
                alpha * (reward + gamma *  np.max(q(new_state)) - q(old_state, action))
        
#       env.render()
        
#        print (q_table[repr(new_state)][action])
        '''
        if i_episodes == N_EPISODES-1:
            print ('i_step:', i_step)
            print ('Done:',done)
            print ('info[actions]:', info['actions'])
            print ('action from choose_actioon:', action)
            print ('reward:', reward, '\n')
        '''
        if done:
            break
        
#    print("Episode finished! Reward: {} | Collisions: {} | Actions: {}".format(reward, info['collisions'], info['actions']))
    
     
    
    if i_episodes == N_EPISODES-1:
        env.render()
        print(f"Episode {i_episodes + 1}: total reward -> {Total_reward}")
