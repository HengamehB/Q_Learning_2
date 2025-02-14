# -*- coding: utf-8 -*-


"""
Implements the 2D Lattice Environment
"""
# Import gym modules
import sys
from math import floor
#from collections import OrderedDict


import gym
from gym import (spaces, utils, logger)
import numpy as np
from six import StringIO
import copy
import scipy

# Human-readable
ACTION_TO_STR = {
    0 : 'L', 1 : 'D',
    2 : 'U', 3 : 'R', 4:'J'}


POLY_TO_INT = { 'x' : 1}


class Lattice2DEnv(gym.Env):
    """A 2-dimensional lattice environment from Dill and Lau, 1989
    [dill1989lattice]_.


    It follows an absolute Cartesian coordinate system, the location of
    the polymer is stated independently from one another. Thus, we have
    four actions (left, right, up, and down) and a chance of collision.


    The environment will first place the initial polymer at the origin. Then,
    for each step, agents place another polymer to the lattice. An episode
    ends when all polymers are placed, i.e. when the length of the action
    chain is equal to the length of the input sequence minus 1. We then
    compute the reward using the energy minimization rule while accounting
    for the collisions and traps.


    Attributes
    ----------
    seq : str
        A chain of lattice vertices each represented by 'x'.
    state : list
        list of the current chain with coordinates.
    actions : list
        List of actions performed by the model.
    grid_length : int
        Length of one side of the grid.
    midpoint : tuple
        Coordinate containing the midpoint of the grid.
    grid : numpy.ndarray
        Actual grid containing the polymer chain.


    .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
    mechanics model of the conformational and sequence spaces of proteins.
    Marcromolecules 22(10), 3986–3997 (1989)
    """
    metadata = {'render.modes': ['human', 'ansi']}
    
    


    def __init__(self, p, collision_penalty= -2, trap_penalty= 0.5):
        """Initializes the lattice


        Parameters
        ----------
        p : str, must only consist of an array of integers.
            Sequence containing the maximum length of each interpolator.
        collision_penalty : int, must be a negative value
            Penalty incurred when the agent made an invalid action.
            Default is -2.
        trap_penalty : float, must be between 0 and 1
            Penalty incurred when the agent is trapped. Actual value is
            computed as :code:`floor(length_of_sequence * trap_penalty)`
            Default is -2.
        """
        
        try:
            if collision_penalty >= 0:
                raise ValueError("%r (%s) must be negative" %
                                 (collision_penalty, type(collision_penalty)))
            if not isinstance(collision_penalty, int):
                raise ValueError("%r (%s) must be of type 'int'" %
                                 (collision_penalty, type(collision_penalty)))
            self.collision_penalty = collision_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'int'" %
                         (collision_penalty, type(collision_penalty)))
            raise


        try:
            if not 0 < trap_penalty < 1:
                raise ValueError("%r (%s) must be between 0 and 1" %
                                 (trap_penalty, type(trap_penalty)))
            self.trap_penalty = trap_penalty
        except TypeError:
            logger.error("%r (%s) must be of type 'float'" %
                         (trap_penalty, type(trap_penalty)))
            raise
            
        
        self.state = [(0, 0)]
        self.master_state = [(0,0)]
        self.actions = []
        self.origin = (0,0)
        self.op_counts = 0
        self.is_looped = False
        
        
        
        # here P is an array with number of Xs allowed for each operator        
        self.p = p
        
        self.seq = ['x'] * p[self.op_counts]
        
        # Grid attributes
        self.grid_length = 4 * len(self.seq) +1
        self.midpoint = (2*len(self.seq), 2*len(self.seq))
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        

        # Automatically assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]


        # Define action-observation spaces
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=int)
        self.last_action = None
    
    
    def step(self, action):
        
        """Updates the current chain with the specified action.


        The action supplied by the agent should be an integer from 0
        to 3. In this case:
            - 0 : left
            - 1 : down
            - 2 : up
            - 3 : right
            - 4 : jump
        The best way to remember this is to note that they are similar to the
        'h', 'j', 'k', and 'l' keys in vim.


        This method returns a set of values similar to the OpenAI gym, that
        is, a tuple :code:`(old_state, self.master_state, reward, done, info, grid)`.


        The observations are arranged as a :code:`numpy.ndarray` matrix, more
        suitable for agents built using convolutional neural networks. The
        'x' is represented as :code:`1`s.
        However, for the actual chain, that is, an :code:`list` and
        not its grid-like representation, can be accessed from
        :code:`info['state_chain]`.


        The reward is calculated at the end of every episode, that is, when
        the length of the chain is equal to the length of the input sequence.


        Parameters
        ----------
        action : int, {0, 1, 2, 3, 4}
            Specifies the position where the next polymer will be placed
            relative to the previous one:
                - 0 : left
                - 1 : down
                - 2 : up
                - 3 : right
                - 4 : jump

        Returns
        -------
        numpy.ndarray
            Current state of the lattice.
        int or None
            Reward for the current episode.
        bool
            Control signal when the episode ends.
        dict
            Additional information regarding the environment.


        Raises
        ------
        AssertionError
            When the specified action is invalid.
        IndexError
            When :code:`step()` is still called even if done signal
            is already :code:`True`.
        """
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))


        self.last_action = action
        #Signals:
        is_trapped = False 
        is_collided = False
        failed_jump = False
        succ_jump = False
        out_of_xs = False

           
        # Obtain coordinate of previous X
        x, y = next(reversed(self.state))
        # Get all adjacent coords and next move based on action
        adj_coords = self._get_adjacent_coords((x, y))  
        
        if action != 4: 
            next_move = adj_coords[action]
          
        # Detects for out_of_xs in the given coordinate
        idx = len(self.state)
        
        
        old_state = copy.deepcopy(self.master_state)        
        
        if set(adj_coords.values()).issubset(self.state):
            logger.warn('Your agent was trapped! Ending the episode.')
            
            is_trapped = True
        
        elif action != 4 and next_move in self.state:

            is_collided = True
        
        
        elif action == 4:
            
            self.op_counts += 1
            
            if self.is_looped:
                
                self.actions.append(action)
                succ_jump = True
                
            else:
                
                failed_jump = True
        else:
            
            self.actions.append(action)
            
            try:
                
                self.seq[idx]
                
                self.state.append(next_move)
                self.master_state.append(next_move)
                
                # Checking for loops
                #########################################
                start_pt = self.state.index(self.origin)
                new_loop = self.state[start_pt:]
        
                self.is_looped =  (4 <= len(new_loop) <= self.p[self.op_counts]) and \
                (self.state[len(self.state)-1] in \
                            self._get_adjacent_coords(self.origin).values())
                #########################################
            
                
            except IndexError:
                
                #logger.error('All sites have been passed! Nowhere left to go!')
                
                out_of_xs = True

        
        grid_3D = self._draw_grid()
        
        #grid_3D = np.repeat(grid[:, :, np.newaxis], 3, axis=2)
        
        #b = scipy.misc.imresize(grid_3D[:,:,0],[self.grid_length,self.grid_length,1],interp='nearest')
        #c = scipy.misc.imresize(grid_3D[:,:,1],[self.grid_length,self.grid_length,1],interp='nearest')
        #d = scipy.misc.imresize(grid_3D[:,:,2],[self.grid_length,self.grid_length,1],interp='nearest')
        
        done  = self.op_counts == (len(self.p))
        
        #print(self.op_counts)
        #grid_3D = np.stack([b,c,d],axis=2)
        
        if not done and (action ==4) : self.change_op()
        
        reward = self._compute_reward(is_trapped, is_collided, done,\
                                      failed_jump, succ_jump, out_of_xs)
        
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped'   : is_trapped,
            'state_chain'  : self.state
        }
        
#        return (old_state, self.master_state, reward, done, info, grid)
        
        
        return (grid_3D, reward, done)
    
    def reset(self):
        """Resets the environment"""
        
        self.state = [(0, 0)]
        self.master_state = [(0,0)]
        self.actions = []
        self.origin = (0,0)
        self.op_counts = 0
        self.is_looped = False
 
        self.seq = ['x'] * self.p[self.op_counts]
        
        # Grid attributes
        self.grid_length = 4 * len(self.seq) +1
        self.midpoint = (2*len(self.seq), 2*len(self.seq))
        
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        

        # Assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]


        # Define action-observation spaces
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=int)

        return self.grid
    
    
    def change_op(self):
        """Changes the operator"""

        self.state = [self.origin] 
        self.master_state.append(self.origin)
        
        self.seq = ['x'] * self.p[self.op_counts]
        
        # Grid attributes
        self.grid_length = 4 * len(self.seq) +1
        self.midpoint = (2*len(self.seq), 2*len(self.seq))
        
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        

        # Assign first element into grid
        self.grid[self.midpoint] = POLY_TO_INT[self.seq[0]]


        # Define action-observation spaces
        self.action_space = spaces.Discrete(5)
        
        self.observation_space = spaces.Box(low=-2, high=1,
                                            shape=(self.grid_length, self.grid_length),
                                            dtype=int)
        
        
    def _get_adjacent_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (x-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        x, y = coords
        adjacent_coords = {
            0 : (x - 1, y),
            1 : (x, y - 1),
            2 : (x, y + 1),
            3 : (x + 1, y),
        }

        return adjacent_coords


    def render(self, mode='human'):
        """Renders the environment"""

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.grid.astype(str)

        # Convert everything to human-readable symbols
        desc[desc == '0'] = '*'
        desc[desc == '1'] = 'x'
        
        # Obtain all x-y indices of elements
        x_free, y_free = np.where(desc == '*')
        x_s, y_s = np.where(desc == 'x')

        # Decode if possible (?)
        desc.tolist()
        try:
            desc = [[c.decode('utf-8') for c in line] for line in desc]
        except AttributeError:
            pass

        # All unfilled spaces are gray
        for unfilled_coords in zip(x_free, y_free):
            desc[unfilled_coords] = utils.colorize(desc[unfilled_coords], "gray")

        # All hydrophobic molecules are bold-green
        for hmol_coords in zip(x_s, y_s):
            desc[hmol_coords] = utils.colorize(desc[hmol_coords], "white")        
        
        # Provide prompt for last action
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Up", "Right", "Jump"]\
                          [self.last_action]))
        else:
            outfile.write("\n")

        # Draw desc
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
    
    def _draw_grid(self):
        """Constructs a grid with the current chain

        Parameters
        ----------
        
        Returns
        -------
        numpy.ndarray
            Grid of shape :code:`(n, n)` with the chain inside
        """
        
        '''
        Why do I need this?
        '''
        self.grid = np.zeros(shape=(self.grid_length, self.grid_length), dtype=int)
        
        state_p = [(2*i, 2*j) for i, j in self.state]
        state_p_copy = copy.deepcopy(state_p)
        
        idx = len(state_p)
        
        for i in range(idx):
            
            x_m, y_m =  tuple (sum(x)/2 for x in zip(state_p[i%idx], state_p[(i+1)%idx]))
            state_p_copy.insert(2*i+1, (int(x_m), int(y_m)))
       
        
        for coord in state_p_copy:
            trans_x, trans_y = tuple(sum(x) for x in zip(self.midpoint, coord))
            # Recall that a numpy array works by indexing the rows first
            # before the columns, that's why we interchange.
            self.grid[(trans_y, trans_x)] = POLY_TO_INT['x']

        #print (state_p_copy)
        
        grid_3D = np.repeat(np.flipud(self.grid)[:, :, np.newaxis], 3, axis=2)
        
        return grid_3D


    def _compute_reward(self, is_trapped, is_collided, done, failed_jump, succ_jump, out_of_xs):
        """Computes the reward for a given time step

        For every timestep, we compute the reward using the following function:

        .. code-block:: python

            reward_t = state_reward 
                       + collision_penalty
                       + actual_trap_penalty
                       + loop_reward
                       + out_of_xs_penalty
                       
        The :code:`state_reward` is only computed at the end of the episode
        (Area_Calc) and its value is :code:`0` for every timestep
        before that.

        The :code:`collision_penalty` is given when the agent makes an invalid
        move, i.e. going to a space that is already occupied.

        The :code:`actual_trap_penalty` is computed whenever the agent
        completely traps itself and has no more moves available. Overall, we
        still compute for the :code:`state_reward` of the current chain but
        subtract that with the following equation:
        :code:`floor(length_of_sequence * trap_penalty)`
        try:

        Parameters
        ----------
        is_trapped : bool
            Signal indicating if the agent is trapped. 
        done : bool
            Done signal
        is_collided : bool
            is_collided signal
        failed_jump : bool
            Signaling when agent was trying to jump without having completed the loop
            of the previous operator.
        succ_jump : bool
            Signaling when agent jumps successfully from one operator to the other. 

        Returns
        -------
        int
            Reward function
        """
        
        '''
        failed_jump_penalty = self.failed_jump_penalty if failed_jump else 0
        + failed_jump_penalty
        '''
        
        collision_penalty = self.collision_penalty if is_collided else 0
        
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0
        
        
        '''
        '''''''''''''''''''''''''''''''''''''''''''''
        ''' This Whole Chunk is Loop Reward '''
        '''''''''''''''''''''''''''''''''''''''''''''
        '''
        if  succ_jump:
            loop_reward = +7
            
        elif failed_jump:
            loop_reward = -5
            
        else:
            loop_reward = 0
      
        out_of_xs_penalty = -10 if out_of_xs else 0
        
        
        reward =  collision_penalty + actual_trap_penalty + loop_reward + out_of_xs_penalty 
        
        return reward


    def _compute_free_energy(self, chain):
        """Computes the Gibbs free energy given the lattice's state


        The free energy is only computed at the end of each episode. This
        follow the same energy function given by Dill et. al.
        [dill1989lattice]_


        Recall that the goal is to find the configuration with the lowest
        energy.


        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and se quence spaces of proteins.
        Marcromolecules 22(10), 3986–3997 (1989)


        Parameters
        ----------
        chain : OrderedDict
            Current chain in the lattice


        Returns
        -------
        int
            Computed free energy
        """
        h_polymers = [x for x in chain if chain[x] == 'H']
        h_pairs = [(x, y) for x in h_polymers for y in h_polymers]


        # Compute distance between all hydrophobic pairs
        h_adjacent = []
        for pair in h_pairs:
            dist = np.linalg.norm(np.subtract(pair[0], pair[1]))
            if dist == 1.0: # adjacent pairs have a unit distance
                h_adjacent.append(pair)


        # Get the number of consecutive H-pairs in the string,
        # these are not included in computing the energy
        h_consecutive = 0
        for i in range(1, len(self.state)):
            if (self.seq[i] == 'H') and (self.seq[i] == self.seq[i-1]):
                h_consecutive += 1


        # Remove duplicate pairs of pairs and subtract the
        # consecutive pairs
        nb_h_adjacent = len(h_adjacent) / 2
        gibbs_energy = nb_h_adjacent - h_consecutive
        reward = - gibbs_energy
        return int(reward)




