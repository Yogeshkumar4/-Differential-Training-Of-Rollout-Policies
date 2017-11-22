from __future__ import print_function
import random

class Environment:
    def __init__(self, side, instance, slipperiness, randomizeNames, seed, maxLength):
        self.side = side
        self.numStates = self.side ** 2
        self.slipperiness = slipperiness
        self.randomizeNames = randomizeNames
        self.maxLength = maxLength
        self.episodeLen = 0
        random.seed(instance)
        self.obstacles = []

        corners = [0, side-1, side*(side-1), side*side-1]
        self.goal = random.choice(corners)
        self.start = random.randint(0, self.numStates - 1)
        while self.start == self.goal:
            self.start = random.randint(0, self.numStates - 1)
        self.reset_start() # Anything but the goal

        # Make some obstacles
        numObstacles = self.numStates // 10
        for i in range(numObstacles):
            obs = random.randint(0, self.numStates - 1)
            while obs == self.start or obs == self.goal:
                obs = random.randint(0, self.numStates - 1)
            self.obstacles.append(obs)


        random.seed(seed)
        # Make a mapping for randomizing state names
        oldnames = list(range(self.numStates))
        newnames = oldnames[:]
        random.shuffle(newnames)
        self.oldToNew = {old: new for old, new in zip(oldnames, newnames)}
        self.newToOld = {new: old for old, new in zip(oldnames, newnames)}

        # Start the environemt
        self.state = self.start

        # self.printWorld()

    def printWorld(self):
        for y in range(self.side):
            print('  |', end='')
            for x in range(self.side):
                state = y * self.side + x
                obs = self.obfuscate(state)
                stateType = ' '
                if state == self.state: stateType = 'A' # This is where the agent is at!
                if state == self.start: stateType = 'S' # Start state
                if state == self.goal: stateType = 'G' # Goal
                if state in self.obstacles: stateType = 'O' # Obstacle
                # print('  {:03} {:03} {}  |'.format(state, obs, stateType), end='')
                print('{}|'.format(stateType), end='')
                
            print()

    def printPolicy(self, Pi):
        for y in range(self.side):
            print('  |', end='')
            chars = [u'\u2191',u'\u2193',u'\u2190', u'\u2192']
            for x in range(self.side):
                state = y* self.side + x
                print('{} |'.format(chars[Pi[state]].encode("utf-8")), end='')
            print()
        print("-"*(self.side)**2)


    def obfuscate(self, state):
        if self.randomizeNames:
            state = self.oldToNew[state]
        return state

    def deobfuscate(self, state):
        if self.randomizeNames:
            state = self.newToOld[state]
        return state

    def getnumStates(self):
        return self.numStates

    def getState(self):
        return self.obfuscate(self.state)

    def step(self, action):
        '''Takes the given action in the current environment
        Returns: (new state, reward, event)'''

        self.episodeLen += 1

        # Simulate slipping
        if random.random() < self.slipperiness:
            action = random.choice('up down left right'.split())

        y, x = self.state // self.side, self.state % self.side

        x_, y_ = x, y
        if action == 'up':
            y_ -= 1
        elif action == 'down':
            y_ += 1
        elif action == 'left':
            x_ -= 1
        elif action == 'right':
            x_ += 1

        state_ = y_ * self.side + x_

        # If we fall out of boundary, or in an obstacle, undo action.
        if (x_ < 0 or x_ >= self.side or y_ < 0 or y_ >= self.side) or state_ in self.obstacles:
            state_ = self.state

        # If we reach the goal, end the episode
        if state_ == self.goal:
            self.episodeLen = 0
            self.state = self.start
            return self.obfuscate(state_), 100, 'goal'
        elif self.episodeLen == self.maxLength:
            self.episodeLen = 0
            self.state = self.start
            return self.obfuscate(state_), -1, 'terminated'
        else:
            self.state = state_
            return self.obfuscate(self.state), -1, 'continue'

    def sampleAction(self, state, action):
        '''Takes the given action at a particular state in the environment
        Doesn't update the environment or the state, just return the result of action on state
        Returns: (new state, reward)!'''

        # Simulate slipping
        if random.random() < self.slipperiness:
            action = random.choice('up down left right'.split())

        y, x = state // self.side, state % self.side

        x_, y_ = x, y
        if action == 'up':
            y_ -= 1
        elif action == 'down':
            y_ += 1
        elif action == 'left':
            x_ -= 1
        elif action == 'right':
            x_ += 1

        state_ = y_ * self.side + x_

        # If we fall out of boundary, or in an obstacle, undo action.
        if (x_ < 0 or x_ >= self.side or y_ < 0 or y_ >= self.side) or state_ in self.obstacles:
            state_ = state

        # If we reach the goal, end the episode
        if state_ == self.goal:
            return self.obfuscate(state_), 100
        else:
            return self.obfuscate(state_), -1

    def reset(self):
        start = random.randint(0, self.numStates - 1)
        while (start == self.goal) or start in self.obstacles:
            start = random.randint(0, self.numStates - 1)
        self.start = start
        self.state = self.start
        self.episodeLen = 0
        return self.start

    def reset_start(self):
        self.state = self.start
        self.episodeLen = 0
        return self.start