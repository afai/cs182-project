import gym
import time
import random
import torch
from torch.autograd import Variable
from NN import *
import numpy as np
import torch.nn.functional as F
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Class for remembering transitions
class ExperienceReplayRAM():
    # Initialize
    def __init__(self, capacity):
        # Remember capacity
        self.capacity = capacity
        # Each row in memory has current state, action, reward, next state, and done
        self.transitions = np.empty((self.capacity, ramSize+1+1+ramSize+1))
        # Remember index for adding and overwriting transitions
        self.index = 0
        # Remember if memory is full
        self.full = False
    # Store a transition
    def store(self, currObs, action, reward, nextObs, done):
        # Current state
        self.transitions[self.index,:ramSize] = currObs.astype("float")
        # Action
        self.transitions[self.index,ramSize] = action
        # Reward
        self.transitions[self.index,ramSize+1] = reward
        # Next state
        self.transitions[self.index,ramSize+2:2*ramSize+2] = nextObs.astype("float")
        # Done
        self.transitions[self.index,-1] = done
        # Increment index
        self.index += 1
        # If we have reached capacity...
        if self.index == self.capacity:
            # Mark as full
            self.full = True
            # Cycle back index
            self.index %= self.capacity
    # Sample a batch of transitions
    def sample(self, batchSize):
        # If we have reached capacity...
        if self.full:
            # Limit is capacity
            limit = self.capacity
        # Else...
        else:
            # Limit is index
            limit = self.index
        # Return sample
        return self.transitions[np.random.choice(limit, size=batchSize, replace=False),:]

# If main...
if __name__ == "__main__":
    # Set parameters
    memoryCapacity = 10000
    numEpisodes = 1000
    closeRender = True
    batchSize = 100
    ramSize = 128
    discount = 0.9
    scores = []
    loss_func = torch.nn.MSELoss()
    # Create environment
    env = gym.make("MsPacman-ram-v0")
    # Create model
    model = NNRAM(input_dim=ramSize, hidden_dim=50, output_dim=env.action_space.n)
    # Create optimizer
    optimizer = torch.optim.RMSprop(model.parameters())
    # Create memory
    memory = ExperienceReplayRAM(memoryCapacity)
    # For each episode...
    for episode in range(numEpisodes):
        # Start the timer
        startTime = time.time()
        # Calculate epsilon
        epsilon = 1.0 - float(episode) / numEpisodes
        # Initialize time-steps and score to zero
        timeSteps = 0
        score = 0
        # Reset environment and set done to false
        currObs = env.reset()
        done = False
        # Render
        env.render(close=closeRender)
        # While game is not done...
        while not done:
            # Compute Q values
            qValues = model.forward(Variable(torch.from_numpy(currObs).unsqueeze(0).float()))
            # If should choose random action...
            if random.random() < epsilon:
                action = env.action_space.sample()
                qValue = qValues.squeeze().data[action]
            # Else...
            else:
                qValue, action = qValues.max(dim=1)
                action = action.data[0]
            # Perform action and observe
            nextObs, reward, done, _ = env.step(action)
            # Store transition
            memory.store(currObs, action, reward, nextObs, done)
            # If enough memory...
            if memory.full or memory.index >= batchSize:
                # Reset gradient
                optimizer.zero_grad()
                # Sample transitions
                trans = memory.sample(batchSize)
                # Separate into parts and wrap in variables
                transQcurr = model.forward(Variable(torch.from_numpy(trans[:,:ramSize]).float()))
                transAction = trans[:,ramSize].astype("int")
                transReward = Variable(torch.from_numpy(trans[:,ramSize+1])).float()
                transQnext = model.forward(Variable(torch.from_numpy(trans[:,ramSize+2:2*ramSize+2]).float()))
                transNotDone = Variable(torch.from_numpy(1-trans[:,-1]).float())
                # Compute prediction
                pred = transQcurr[np.arange(batchSize),transAction]
                # Compute target
                target = transReward + discount * (transNotDone * transQnext.max(dim=1)[0])
                # Calculate loss
                loss = loss_func(pred, Variable(target.data, requires_grad=False))
                # Backward propogation
                loss.backward()
                # Update parameters
                optimizer.step()
            # Render
            env.render(close=closeRender)
            # Update score and time
            currObs = nextObs
            score += reward
            timeSteps += 1
        # Print timesteps, score
        print "Episode {0:>3}: {1:>4} steps, score {2:>6}, time {3:>6.2f}".format(episode, timeSteps+1, int(score), time.time()-startTime)
        # Store score
        scores.append(score)
    # Plot scores
    plt.plot(scores)
    plt.savefig("scores.png")
    # Save model
    with open("DQNpickle/MsPacman-ram.pkl", "wb") as f:
        pickle.dump(model, f)