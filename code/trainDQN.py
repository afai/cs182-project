import sys, time, random
import gym
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from NN import *
from ExperienceReplay import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Global parameters
MEMORY_CAPACITY = 50000
BATCH_SIZE = 32
NUM_FRAMES = 4
CONVOLUTION_BRANCH = 2
MODEL_UPDATE = 25
EPSILON_MIN = 0.1
EPSILON_STOP = 0.7
GRAY_WEIGHTS = [0.3, 0.6, 0.1]

# Function to convert to grayscale and reduce the size of the image
def processObs(img, isRAM):
    # If RAM...
    if isRAM:
        # Do nothing
        return img.astype("uint8")
    # Else...
    else:
        # Grey and down-sample
        return np.average(img[::2, ::2], axis=2, weights=GRAY_WEIGHTS).astype("uint8")

# If main...
if __name__ == "__main__":
    # Set parameters
    game = "Breakout"
    isRAM = False
    numEpisodes = 1000
    closeRender = True
    oneLife = True
    discount = 0.99
    loss_func = torch.nn.SmoothL1Loss()
    # Initialize environment and get observation dimensions
    env = gym.make(game + isRAM * "-ram" + "-v4")
    obsDims = processObs(env.reset(), isRAM).shape
    tileDims = tuple([NUM_FRAMES] + len(obsDims) * [1])
    # Specify memory arguments and initialize memory
    memoryArgs = {"capacity": MEMORY_CAPACITY,
                  "batchSize": BATCH_SIZE,
                  "numFrames": NUM_FRAMES,
                  "obsDims": obsDims}
    memory = ExperienceReplay(**memoryArgs)
    # Specify model arguments and initialize models
    modelType = NNRAM if isRAM else NNscreen
    modelArgs = {"num_frames": NUM_FRAMES,
                 "b": CONVOLUTION_BRANCH,
                 "input_dim": obsDims,
                 "output_dim": env.action_space.n}
    modelOpt = modelType(**modelArgs)
    modelFix = modelType(**modelArgs)
    # Initialize optimizers
    optimizer = torch.optim.Adam(modelOpt.parameters(), )
    # Store scores and loss
    scores = []
    losses = []
    # Start the training timer
    startTime = time.time()
    # For each episode...
    for episode in range(numEpisodes):
        # If epsilon has stopped decreasing...
        # if float(episode) / numEpisodes >= EPSILON_STOP:
        #     # Render
        #     closeRender = False
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
        currObsStacked = np.tile(processObs(env.reset(), isRAM), tileDims)
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
                # Compute Q values using optimized model
                qValues = modelOpt.forward(Variable(torch.from_numpy(currObsStacked).unsqueeze(0).float()))
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
            nextObsStacked = np.vstack((currObsStacked[1:], np.expand_dims(processObs(nextObs, isRAM), 0)))
            # Store transition
            memory.store(currObsStacked, nextObsStacked, action, reward, done)
            # If enough memory
            if memory.full or memory.index >= BATCH_SIZE:
                # Reset gradient
                optimizer.zero_grad()
                # Sample transitions
                transSTS, transARD = memory.sample(BATCH_SIZE)
                # Separate into parts and wrap in variables
                transScurr = Variable(torch.from_numpy(transSTS[:,:,:obsDims[0]]).float())
                transSnext = Variable(torch.from_numpy(transSTS[:,:,obsDims[0]:2*obsDims[0]]).float())
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
    plt.savefig("plots/" + game + isRAM * "-ram" + "_" + str(numEpisodes) + "e_scores.png")
    plt.close()
    # Plot loss and save
    plt.plot(losses)
    plt.savefig("plots/" + game + isRAM * "-ram" + "_" + str(numEpisodes) + "e_losses.png")
    plt.close()
    # Save model
    with open("models/" + isRAM * "RAM/" + (not isRAM) * "screen/" + game, "wb") as f:
        torch.save(modelOpt, f)