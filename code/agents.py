import Queue, copy, random
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from NN import *

# Agent class
class Agent:
    # Must have a method to get action(s)
    def getActions(self, env, obs):
        raise Exception('Not Defined!')

# Random agent
class RandomAgent(Agent):
    # Return a random action
    def getActions(self, env, obs):
        return [env.action_space.sample()]

# BFS agent
class BFSAgent(Agent):
    # Initialize
    def __init__(self, maxActions=1, oneAction=True):
        self.maxActions = maxActions
        self.oneAction = oneAction
    # Perform BFS
    def getActions(self, env, obs):
        # Get the current state
        currentState = env.env.clone_state()
        # Create a fringe
        fringe = Queue.Queue()
        # Add the current state and an empty path
        fringe.put((currentState, [], 0, False))
        # Record best path and maximum reward
        maxPath = None
        maxReward = None
        maxDone = None
        # While the fringe is not empty
        while not fringe.empty():
            # Pop a state and path
            popState, popPath, popReward, popDone = fringe.get()
            # If maximum actions planned or done...
            if len(popPath) == self.maxActions or popDone:
                # If no max or higher reward...
                if maxReward == None or (popReward > maxReward and not (popDone and not maxDone)):
                    # Set it as max
                    maxPath = popPath
                    maxReward = popReward
                    maxDone = popDone
            # Else...
            else:
                # For each action...
                for stepAction in range(env.action_space.n):
                    # Restore the environment with the state
                    env.env.restore_state(popState)
                    # Step with the action
                    _, stepReward, stepDone, _ = env.step(stepAction)
                    # Add successor to fringe
                    fringe.put((env.env.clone_state(), popPath + [stepAction], popReward + stepReward, stepDone))
        # Restore environment with the current state
        env.env.restore_state(currentState)
        # If only one action...
        if self.oneAction:
            # Truncate path
            maxPath = maxPath[:1]
        # Return actions
        return maxPath

class DQNRAMagent(Agent):
    # Initialize
    def __init__(self, game, epsilon=0.0):
        with open("DQNmodels/" + game, "rb") as f:
            self.model = torch.load(f)
        self.epsilon = epsilon
    # Get best action
    def getActions(self, env, obs):
        # IF the stacked observation does not exist...
        if not hasattr(self, 'obsStacked'):
            # Create it
            self.obsStacked = np.tile(obs, (self.model.num_frames, 1))
        # Update stacked observation
        self.obsStacked = np.vstack((self.obsStacked[1:], np.expand_dims(obs, 0)))
        # If should choose random action...
        if random.random() < self.epsilon:
            # Choose random action
            action = [env.action_space.sample()]
        # Else...
        else:
            # Compute Q values
            qValues = self.model.forward(Variable(torch.from_numpy(self.obsStacked).unsqueeze(0).float()))
            # Select best action
            qValue, action = qValues.max(dim=1)
            action = action.data[0]
        # Return action
        return [action]
