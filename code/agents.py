import gym
import Queue
import copy
from NN import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

# Agent class
class Agent:
    # Must have a method to get action(s)
    def getActions(self, env):
        raise Exception('Not Defined!')

# Random agent
class RandomAgent(Agent):
    # Return a random action
    def getActions(self, env):
        return [env.action_space.sample()]

# BFS agent
class BFSAgent(Agent):
    # Initialize
    def __init__(self, maxActions=1, oneAction=True):
        self.maxActions = maxActions
        self.oneAction = oneAction
    # Perform BFS
    def getActions(self, env):
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
    def __init__(self, game):
        with open("DQNpickle/"+game+".pkl", "rb") as f:
            self.model = pickle.load(f)
    # Get best action
    def getActions(self, env):
        # Get current observation
        obs = env.env.ale.getRAM()
        # Compute A values
        qValues = self.model.forward(Variable(torch.from_numpy(obs).unsqueeze(0).float()))
        # Compute best action
        qValue, action = qValues.max(dim=1)
        action = action.data[0]
        # Return action
        return [action]
