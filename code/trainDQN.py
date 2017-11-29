import sys, time, random
import gym
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from NN import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Global parameters
MEMORY_CAPACITY = 10000
BATCH_SIZE = 100
RAM_SIZE = 128
HIDDEN_DIM = 50

# Class for remembering transitions
class ExperienceReplayRAM():
    # Initialize
    def __init__(self, capacity):
        # Remember capacity
        self.capacity = capacity
        # Each row in memory has current state, action, reward, next state, and done
        self.transitions = np.empty((self.capacity, RAM_SIZE+1+1+RAM_SIZE+1))
        # Remember index for adding and overwriting transitions
        self.index = 0
        # Remember if memory is full
        self.full = False
    # Store a transition
    def store(self, currObs, action, reward, nextObs, done):
        # Current state
        self.transitions[self.index,:RAM_SIZE] = currObs.astype("float")
        # Next state
        self.transitions[self.index,RAM_SIZE:2*RAM_SIZE] = nextObs.astype("float")
        # Action
        self.transitions[self.index,-3] = action
        # Reward
        self.transitions[self.index,-2] = reward
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
    def sample(self, BATCH_SIZE):
        # If we have reached capacity...
        if self.full:
            # Limit is capacity
            limit = self.capacity
        # Else...
        else:
            # Limit is index
            limit = self.index
        # Return sample
        return self.transitions[np.random.choice(limit, size=BATCH_SIZE, replace=False),:]

# If main...
if __name__ == "__main__":
    # Set parameters
    game = "MsPacman"
    numEpisodes = 1000
    closeRender = False
    discount = 0.9
    loss_func = torch.nn.MSELoss()
    # Initialize environment, model, optimizer, and memory
    env = gym.make(game + "-ram-v0")
    model = NNRAM(input_dim=RAM_SIZE, hidden_dim=HIDDEN_DIM, output_dim=env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters())
    memory = ExperienceReplayRAM(MEMORY_CAPACITY)
    # Store scores and loss
    scores = []
    losses = []
    # For each episode...
    for episode in range(numEpisodes):
        # Start the timer
        startTime = time.time()
        # Calculate epsilon
        epsilon = 1.0 - float(episode) / numEpisodes / PERCENT_TRAIN
        # Initialize time-steps, score, and loss
        timeSteps = 0
        score = 0.0
        episodeLoss = 0.0
        numSamples = 0
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
            # If enough memory and still training
            if memory.full or memory.index >= BATCH_SIZE:
                # Reset gradient
                optimizer.zero_grad()
                # Sample transitions
                trans = memory.sample(BATCH_SIZE)
                # Separate into parts and wrap in variables
                transScurr = Variable(torch.from_numpy(trans[:,:RAM_SIZE]).float())
                transSnext = Variable(torch.from_numpy(trans[:,RAM_SIZE:2*RAM_SIZE]).float())
                transAction = trans[:,-3].astype("int")
                transReward = Variable(torch.from_numpy(trans[:,-2]).float())
                transNotDone = Variable(torch.from_numpy(1-trans[:,-1]).float())
                # Compute Q values
                transQcurr = model.forward(transScurr)
                transQnext = model.forward(transSnext)
                # Compute input
                pred = transQcurr[np.arange(BATCH_SIZE),transAction]
                # Compute target
                target = transReward + discount * (transNotDone * transQnext.max(dim=1)[0])
                # Calculate loss, unwrapping target vector
                loss = loss_func(pred, Variable(target.data))
                # Backward propogation
                loss.backward()
                # Add to loss
                episodeLoss += loss.data[0]
                numSamples += 1
                # Update parameters
                optimizer.step()
            # Render
            env.render(close=closeRender)
            # Update score and time
            currObs = nextObs
            score += reward
            timeSteps += 1
        # Average loss
        episodeLoss /= (numSamples + 1)
        # Print timesteps, score
        print "Episode {0:>3}: {1:>4} steps, score {2:>6}, time {3:>6.2f}, loss {4:>6.2f}".format(episode, timeSteps+1, int(score), time.time()-startTime, episodeLoss)
        # Store score and loss
        scores.append(score)
        losses.append(episodeLoss)
    # Plot scores and save
    plt.plot(scores)
    plt.savefig("plots/" + game + "_" + str(numEpisodes) + "e_scores.png")
    plt.close()
    # Plot loss and save
    plt.plot(losses)
    plt.savefig("plots/" + game + "_" + str(numEpisodes) + "e_losses.png")
    plt.close()
    # Save model
    with open("DQNmodels/" + game + "-ram", "wb") as f:
        torch.save(model, f)