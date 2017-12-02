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
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
RAM_SIZE = 128
NUM_FRAMES = 4
CONVOLUTION_BRANCH = 2
MODEL_UPDATE = 25
EPSILON_MIN = 0.1
EPSILON_STOP = 0.7

# Class for remembering transitions
class ExperienceReplayRAM():
    # Initialize
    def __init__(self, capacity):
        # Remember capacity
        self.capacity = capacity
        # Each row in memory has current state, next state, action, reward, and done
        self.transSTS = np.empty((self.capacity, NUM_FRAMES, 2*RAM_SIZE), dtype="uint8")
        self.transARD = np.empty((self.capacity, 3), dtype="uint8")
        # Remember index for adding and overwriting transitions
        self.index = 0
        # Remember if memory is full
        self.full = False
    # Store a transition
    def store(self, currObsStacked, nextObsStacked, action, reward, done):
        # Create transition
        sts = np.hstack((currObsStacked, nextObsStacked))
        ard = [action, reward, done]
        # Store transition
        self.transSTS[self.index] = sts
        self.transARD[self.index] = ard
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
        # Set limit
        limit = max(self.index, self.full * self.capacity)
        # Generate random indices
        sampleInds = np.random.choice(limit, size=BATCH_SIZE, replace=False)
        # Return sample
        return self.transSTS[sampleInds], self.transARD[sampleInds]

# If main...
if __name__ == "__main__":
    # Set parameters
    game = "Breakout"
    numEpisodes = 100000
    closeRender = True
    oneLife = True
    discount = 0.99
    loss_func = torch.nn.SmoothL1Loss()
    # Initialize environment, model, optimizer, and memory
    env = gym.make(game + "-ram-v4")
    modelOpt = NNRAM(num_frames=NUM_FRAMES, b=CONVOLUTION_BRANCH, input_dim=RAM_SIZE, output_dim=env.action_space.n)
    modelFix = NNRAM(num_frames=NUM_FRAMES, b=CONVOLUTION_BRANCH, input_dim=RAM_SIZE, output_dim=env.action_space.n)
    optimizer = torch.optim.Adam(modelOpt.parameters(), lr=0.0001)
    memory = ExperienceReplayRAM(MEMORY_CAPACITY)
    # Store scores and loss
    scores = []
    losses = []
    # Start the training timer
    startTime = time.time()
    # For each episode...
    for episode in range(numEpisodes):
        # If epsilon has stopped decreasing...
        if float(episode) / numEpisodes >= EPSILON_STOP:
            # Render
            closeRender = False
        # If we have performed 25 episodes...
        if episode % 25 == 0:
            # Update fixed model
            modelFix.load_state_dict(modelOpt.state_dict())
        # Calculate epsilon
        epsilon = 1.0 - (1.0 - EPSILON_MIN) / EPSILON_STOP * (float(episode) / numEpisodes)
        epsilon = max(epsilon, EPSILON_MIN)
        # Initialize time-steps, score, and loss
        timeSteps = 0
        score = 0.0
        episodeLoss = 0.0
        numSamples = 0
        # Reset environment and set done to false
        currObsStacked = np.tile(env.reset(), (NUM_FRAMES, 1))
        lives = env.env.ale.lives()
        done = False
        # Render
        env.render(close=closeRender)
        # While game is not done...
        while not done:
            # If should choose random action...
            if random.random() < epsilon:
                # Choose random action
                action = env.action_space.sample()
            # Else...
            else:
                # Compute Q values using fixed model
                qValues = modelFix.forward(Variable(torch.from_numpy(currObsStacked).unsqueeze(0).float()))
                # Select best action
                qValue, action = qValues.max(dim=1)
                action = action.data[0]
            # Perform action and observe
            nextObs, reward, done, info = env.step(action)
            # If one life per episode and we lost it...
            if oneLife and lives > info["ale.lives"]:
                # Set done to True to end episode
                done = True
            # Compute next observation stacked
            nextObsStacked = np.vstack((currObsStacked[1:], np.expand_dims(nextObs, 0)))
            # Store transition
            memory.store(currObsStacked, nextObsStacked, action, reward, done)
            # If enough memory
            if memory.full or memory.index >= BATCH_SIZE:
                # Reset gradient
                optimizer.zero_grad()
                # Sample transitions
                transSTS, transARD = memory.sample(BATCH_SIZE)
                # Separate into parts and wrap in variables
                transScurr = Variable(torch.from_numpy(transSTS[:,:,:RAM_SIZE]).float())
                transSnext = Variable(torch.from_numpy(transSTS[:,:,RAM_SIZE:2*RAM_SIZE]).float())
                transAction = transARD[:,-3].astype("int")
                transReward = Variable(torch.from_numpy(transARD[:,-2]).float())
                transNotDone = Variable(torch.from_numpy(1-transARD[:,-1]).float())
                # Compute Q values from appropriate model
                transQcurr = modelOpt.forward(transScurr)
                transQnext = modelFix.forward(transSnext)
                # Compute prediction
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
            currObsStacked = nextObsStacked
            score += reward
            timeSteps += 1
        # Average loss
        episodeLoss /= numSamples
        # Print timesteps, score
        print "Ep {0:>6}: steps {1:>6}, score {2:>6}, time {3:>9.2f}, loss {4:>10.2f}".format(episode, timeSteps+1, int(score), time.time()-startTime, episodeLoss)
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
        torch.save(modelOpt, f)