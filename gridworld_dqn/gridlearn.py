from gridworld import *
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

N = 5

env = GridWorldEnv(partial=False, size=N)
plt.show()

class QNet():
    def __init__(self, h_size):
        self.frame_input = tf.placeholder(
            shape=[None, IMG_SIZE*IMG_SIZE*3], dtype=tf.float32)
        self.frame = tf.reshape(self.frame_input, shape=[-1,IMG_SIZE,IMG_SIZE,3])

        # TODO(gkanwar): What does num_outputs dictate?
        self.conv = []
        self.conv.append(slim.conv2d(
            inputs=self.frame_input, num_outputs=32, kernel_size=[8,8],
            stride=[4,4], padding='VALID', biases_initializer=None))
        self.conv.append(slim.conv2d(
            inputs=self.conv[-1], num_outputs=64, kernel_size=[4,4],
            stride=[2,2], padding='VALID', biases_initializer=None))
        self.conv.append(slim.conv2d(
            inputs=self.conv[-1], num_outputs=64, kernel_size=[3,3],
            stride=[1,1], padding='VALID', biases_initializer=None))
        self.conv.append(slim.conv2d(
            inputs=self.conv[-1], num_outputs=64, kernel_size=[7,7],
            stride=[1,1], padding='VALID', biases_initializer=None))
        
        # Advantage and value streams (dualing Q learning)
        self.stream_ac, self.stream_vc = tf.split(self.conv[-1], 2, 3)
        self.stream_a = slim.flatten(self.stream_ac)
        self.stream_v = slim.flatten(self.stream_vc)
        # Xavier init to have best possible signal penetration
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.a_W = tf.Variable(xavier_init([h_size//2, env.n_actions]))
        self.v_W = tf.Variable(xavier_init([h_size//2, 1]))
        self.a_out = tf.matmul(self.stream_a, a_W)
        self.v_out = tf.matmul(self.stream_v, v_W)

        # Combining streams
        a_meanless = tf.subtract(
            self.a_out, tf.reduce_mean(self.a_out, axis=1, keep_dims=True))
        self.Q_out = self.v_out + a_meanless
        self.Q_predict = tf.argmax(self.Q_out, 1)

        # Loss fn
        self.Q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(
            self.actions, env.n_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(
            self.Q_out, self.actions_onehot), axis=1)

        self.batch_err = tf.square(self.Q_target - self.Q)
        self.loss = tf.reduce_mean(self.batch_err)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = self.trainer.minimize(self.loss)

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
            np.array(random.sample(self.buf, size)), [size, exp_size])

# Just flatten for input to network
def preprocess_state(state):
    return np.reshape(state, [IMG_SIZE*IMG_SIZE*3])

# Update target network using main network
def make_update_target(tf_vars, tau):
    n_vars = len(tf_vars)
    ops = []
    for i,var in enumerate(tf_vars[0:n_vars//2]):
        ops.append(tf_vars[i+n_vars//2].assign(
            tau*var.value() + (1-tau)*tf_vars[i+n_vars//2].value()))
    return ops
def run_ops(ops, sess):
    for op in ops: sess.run(op)

# epsilon-greedy action selection from range(n)
def eps_greedy(blind, greedy, eps, pre_train=False):
    if pre_train or np.random.rand(1) < eps:
        return blind()
    else:
        return greedy()


### Do the thing!
batch_size = 32
update_freq = 4  # frequency of training steps
gamma = 0.99  # discount
start_eps = 1.0
end_eps = 0.1
annealing_steps = 100
num_episodes = 100
pre_train_steps = 100
max_episode_len = 50
load_model = False
model_path = "./gridmodel/"
h_size = 512
tau = 0.001

tf.reset_default_graph()
main_net = QNet(h_size)
target_net = QNet(h_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
update_target_ops = make_update_target(trainables, tau)
global_exp_buf = ExperienceBuffer()

eps = start_eps
delta_eps = (start_eps - end_eps) / annealing_steps
total_steps = 0 # counter for pre-training

all_rewards = []

def update_eps():
    global eps
    if total_steps > pre_train_steps and eps > end_eps:
        eps -= delta_eps

if not os.path.exists(path): os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading ' + model_path)
        checkpoint = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    for i in range(num_episodes):
        episode_exp_buf = ExperienceBuffer()
        state = preprocess_state(env.reset())
        done = False
        episode_reward = 0
        def blind(): return np.random.randint(env.n_actions)
        def greedy(): return sess.run(
                main_net.predict, feed_dict={main_net.frame_input: [state]})[0]
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
                # Double DQN
                Q1 = sess.run(main_net.Q_predict, feed_dict={
                    main_net.frame_input: np.vstack(train_batch[:,3])})
                Q2 = sess.run(target_net.Q_out, feed_dict={
                    target_net.frame_input: np.vstack(train_batch[:,3])})
                done_mask = 1 - train_batch[:,4] # 0 = done, 1 = not done
                print('Q2 dims: ' + str(Q2.shape))
                assert(Q2.shape[0] == batch_size) # ??
                QQ = Q2[range(batch_size), Q1]
                train_reward = train_batch[:,2]
                Q_target = train_reward + gamma * QQ * done_mask
                sess.run(main_net.update_model, feed_dict = {
                    main_net.frame_input: np.vstack(train_batch[:,0]),
                    main_net.Q_target: Q_target,
                    main_net.actions: train_batch[:,1]})
                run_ops(update_target_ops, sess)

            state = new_state
            if done: break

        global_exp_buf.add(episode_exp_buf.buf)
        all_rewards.append(episode_reward)
        if i % 1000 == 0:
            filename = path+'/model-'+str(i)+'.mdl'
            saver.save(sess, filename)
            print('Saved model ' + filename)
        if len(all_rewards) % 10 == 0:
            print(total_steps, np.mean(all_rewards[-10:]), eps)
    filename = path+'/model-final.mdl'
    saver.save(sess, filename)

print("Percent successful episodes: " + str(sum(all_rewards) / num_episodes))
