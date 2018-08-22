"""
NOTE(gkanwar): Source is github.com/awjuliani/DeepRL-Agents.
"""

import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt


class GridObj():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class GridWorldEnv():
    actions = [
        'up',
        'down',
        'left',
        'right'
    ]
    
    def __init__(self, partial, size):
        self.IMG_SIZE = size + 2 # account for border
        self.sizeX = size
        self.sizeY = size
        self.n_actions = len(self.actions)
        self.objects = []
        self.partial = partial
        a = self.reset()
        plt.imshow(a,interpolation="nearest")
        
        
    def reset(self):
        self.objects = []
        hero = GridObj(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        goal = GridObj(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal)
        guard = GridObj(self.newPosition(),1,1,0,-1,'guard')
        self.objects.append(guard)
        # goal2 = GridObj(self.newPosition(),1,1,1,1,'goal')
        # self.objects.append(goal2)
        # guard2 = GridObj(self.newPosition(),1,1,0,-1,'guard')
        # self.objects.append(guard2)
        # goal3 = GridObj(self.newPosition(),1,1,1,1,'goal')
        # self.objects.append(goal3)
        # goal4 = GridObj(self.newPosition(),1,1,1,1,'goal')
        # self.objects.append(goal4)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self, direction):
        d = self.actions[direction]
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if d == 'up' and hero.y >= 1:
            hero.y -= 1
        if d == 'down' and hero.y <= self.sizeY-2:
            hero.y += 1
        if d == 'left' and hero.x >= 1:
            hero.x -= 1
        if d == 'right' and hero.x <= self.sizeX-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY) ]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GridObj(self.newPosition(),1,1,1,1,'goal'))
                else: 
                    self.objects.append(GridObj(self.newPosition(),1,1,0,-1,'guard'))
                return other.reward,False
        if ended == False:
            return 0.0,False

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = scipy.misc.imresize(a[:,:,0],[self.IMG_SIZE,self.IMG_SIZE,1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[self.IMG_SIZE,self.IMG_SIZE,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[self.IMG_SIZE,self.IMG_SIZE,1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        return a

    ### Returns: (state, reward, done)
    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done
        else:
            return state,(reward+penalty),done
