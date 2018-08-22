from gridworld import *
from collections import defaultdict
import os
import pickle
import random
import numpy as np
import sys
import time

N = 5
env = GridWorldEnv(partial=False, size=N)

class ExperienceBuffer():
    exp_template = ('state', 'action', 'reward', 'new_state', 'done')
    exp_size = len(exp_template)
    def __init__(self, buf_size=50000):
        self.buf = []
        self.buf_size = buf_size

    def add(self, exp):
        # Clear out space at the beginning of buf if needed
        excess = len(self.buf) + len(exp) - self.buf_size
        if excess > 0:
            self.buf[0:excess] = []
        self.buf.extend(exp)

    def sample(self, size):
        return np.reshape(
            np.array(random.sample(self.buf, size)), [size, self.exp_size])

# Convert to flat tuple so it can be hashed
def preprocess_state(state):
    return tuple(np.reshape(state, [env.IMG_SIZE*env.IMG_SIZE*3]))

# epsilon-greedy action selection
def eps_greedy(blind, greedy, eps, pre_train=False):
    if pre_train or np.random.rand(1) < eps:
        return blind()
    else:
        return greedy()

batch_size = 32
update_freq = 8  # frequency of training steps
gamma = 0.99  # discount
start_eps = 1.0
end_eps = 0.1
annealing_steps = 10000
num_episodes = 10000
pre_train_steps = 10000
max_episode_len = 50
save_path = "./gridmodel_basic"
alpha = 0.01

# Big Q-function as a dictionary, mapping state -> action_dict,
# where action_dict maps action -> value.
Q = defaultdict(dict) # default is empty action_dict

def blind(): return np.random.randint(env.n_actions)
DEFAULT_VALUE = -10
def get_best_act(act_dict):
    best_act = None
    best_val = None
    for act in act_dict:
        val = act_dict[act]
        if best_val is None or val > best_val:
            best_val = val
            best_act = act
    if best_val is None:
        return (blind(), DEFAULT_VALUE)
    else:
        return (best_act, best_val)

def update_Q(exp):
    state, action, reward, new_state, done = exp
    if not done:
        predict_act, predict_val = get_best_act(Q[new_state])
    else:
        predict_val = 0
    target = reward + gamma*predict_val
    if action in Q[state]:
        Q[state][action] = (1-alpha)*Q[state][action] + alpha*target
    else:
        Q[state][action] = target
    

global_exp_buf = ExperienceBuffer()
    
eps = start_eps
delta_eps = (start_eps - end_eps) / annealing_steps
total_steps = 0 # counter for pre-training

all_rewards = []

def update_eps():
    global eps
    if total_steps > pre_train_steps and eps > end_eps:
        eps -= delta_eps

if not os.path.exists(save_path): os.makedirs(save_path)

for i in range(num_episodes):
    episode_exp_buf = ExperienceBuffer()
    state = preprocess_state(env.reset())
    done = False
    episode_reward = 0
    def greedy():
        (best_act, best_val) = get_best_act(Q[state])
        return best_act
    for j in range(max_episode_len):
        action = eps_greedy(blind, greedy, eps, total_steps < pre_train_steps)
        new_state, reward, done = env.step(action)
        new_state = preprocess_state(new_state)
        total_steps += 1
        episode_reward += reward
        episode_exp_buf.add(np.reshape(
            np.array([state, action, reward, new_state, done]), [1,5]))
        update_eps()

        if total_steps > pre_train_steps and total_steps % update_freq == 0:
            train_batch = global_exp_buf.sample(batch_size)
            for exp in train_batch:
                update_Q(exp)

        state = new_state
        if done: break

    global_exp_buf.add(episode_exp_buf.buf)
    all_rewards.append(episode_reward)
    if i % 1000 == 0:
        filename = save_path+'/model-'+str(i)+'.mdl'
        with open(filename, 'wb') as f:
            pass
            # pickle.dump(Q, f)
        print('Saved model ' + filename)
    if len(all_rewards) % 10 == 0:
        print(total_steps, np.mean(all_rewards[-10:]), eps)
    filename = save_path+'/model-final.mdl'
    with open(filename, 'wb') as f:
        pass
        # pickle.dump(Q, f)

with open(save_path+'/rewards.pkl', 'wb') as f:
    pickle.dump(all_rewards, f)
print("Avg episode reward: " + str(sum(all_rewards) / num_episodes))
