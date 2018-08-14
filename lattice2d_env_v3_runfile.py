#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Sun Jul 15 15:52:28 2018
    
    @author: Hengameh
    """

from gym_lattice.envs import Lattice2DEnv
from gym import spaces
import numpy as np
import random

np.random.seed(42)

p = [8,4,6,6] # number and length of operators
action_space = spaces.Discrete(5) # Choose among [0, 1, 2 , 3, 4]
N_EPISODES = 1000
MAX_EPISODE_STEPS = sum(p)
env = Lattice2DEnv(p)

MIN_ALPHA = 0.0001
alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)

MIN_EPSILON = 0.0
epsilons = np.linspace(0.9, MIN_EPSILON, N_EPISODES)
gamma = 0.95

q_table = dict()

################# Q Learning Functions #################

def choose_action(state,eps):
    if random.uniform(0, 1) < eps:
        
        action = action_space.sample()
        
        return  action
    else:
        return np.argmax(q(state))

def q(state, action=None):
    
    state = repr(state)
    
    if state not in q_table:
        q_table[state] = np.zeros(5)
    
    if action is None:
        return q_table[state]
    
    return q_table[state][action]

######################## Tester ########################

'''
    env.reset()
    old_state, new_state, reward, done, info, grid = env.step(0)
    
    print(reward)
    
    old_state, new_state, reward, done, info, grid = env.step(1)
    
    print(reward)
    
    old_state, new_state, reward, done, info, grid = env.step(3)
    
    print(reward)
    
    #############
    old_state, new_state, reward, done, info, grid = env.step(4)
    
    print (reward)
    #############
    
    '''

for i_episodes in range(N_EPISODES):
    
    #    print (i_episodes)
    
    env.reset()
    Total_reward = 0
    alpha = alphas[i_episodes]
    eps = epsilons[i_episodes]
    
    while True:
        
        # Sampleing from the action space in an epsilon-greedy manner
        action = choose_action(env.state,eps)
        
        old_state, new_state, reward, done, info, grid = env.step(action)
        
        Total_reward += reward
        
        q(old_state)[action] = q(old_state, action) + \
            alpha * (reward + gamma *  np.max(q(new_state)) - q(old_state, action))
        
        if done:
            break

print(f"Episode {i_episodes + 1}: total reward -> {Total_reward}, action -> {info['actions']}")
if i_episodes == N_EPISODES-1:
    env.render()





