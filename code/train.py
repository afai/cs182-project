import sys, time, random, pickle
import gym
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
from NN import *
from ExperienceReplay import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#########################
### UTILITY FUNCTIONS ###
########################

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


##########################
### TRAINING FUNCTIONS ###
##########################

# Function to train DQN with experience replay
def trainDQNep(modelOpt, modelFix, scores, losses, episode):
    # Create separate environment
    env = gym.make(gameString)
    # Specify memory arguments and initialize memory
    memoryArgs = {"capacity": MEMORY_CAPACITY,
                  "batchSize": BATCH_SIZE,
                  "numFrames": NUM_FRAMES,
                  "obsDims": obsDims}
    memory = ExperienceReplay(**memoryArgs)
    # For each episode...
    while episode.value < numEpisodes:
        # Get current episode and increment tracked episode
        with episode.get_lock():
            ep = episode.value
            episode.value += 1
        # If we have performed 25 episodes...
        if ep % 25 == 0:
            # Update fixed model
            modelFix.load_state_dict(modelOpt.state_dict())
        # Calculate epsilon
        epsilon = 1.0 - (1.0 - EPSILON_MIN) / EPSILON_STOP * (float(ep) / numEpisodes)
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
        episodeLoss /= (numSamples + 1)
        # Print timesteps, score
        print EPISODE_STRING.format(ep, timeSteps+1, int(score), time.time()-startTime, episodeLoss)
        # Store score and loss
        scores[ep] = score
        losses[ep] = episodeLoss

# Function to train DQN asynchronously
def trainDQNasync(modelOpt, modelFix, scores, losses, episode):
    # Create separate environment
    env = gym.make(gameString)
    # For each episode...
    while episode.value < numEpisodes:
        # Get current episode and increment tracked episode
        with episode.get_lock():
            ep = episode.value
            episode.value += 1
        # If we have performed 25 episodes...
        if ep % 25 == 0:
            # Update fixed model
            modelFix.load_state_dict(modelOpt.state_dict())
        # Calculate epsilon
        epsilon = 1.0 - (1.0 - EPSILON_MIN) / EPSILON_STOP * (float(ep) / numEpisodes)
        epsilon = max(epsilon, EPSILON_MIN)
        # Initialize time-steps, score, and loss
        timeSteps = 0
        score = 0.0
        # Reset environment and set done to false
        currObsStacked = np.tile(processObs(env.reset(), isRAM), tileDims)
        lives = env.env.ale.lives()
        done = False
        # Render
        env.render(close=closeRender)
        # Track observations, actions, rewards, and terminal
        currObsAll = []
        nextObsAll = []
        actionsAll = []
        rewardsAll = []
        doneAll = []
        # Reset gradient
        optimizer.zero_grad()
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
            # Render
            env.render(close=closeRender)
            # Add to lists
            currObsAll.append(np.expand_dims(currObsStacked, 0))
            nextObsAll.append(np.expand_dims(nextObsStacked, 0))
            actionsAll.append(action)
            rewardsAll.append(reward)
            doneAll.append(done)
            # Update score and time
            currObsStacked = nextObsStacked
            score += reward
            timeSteps += 1
        # Turn to numpy arrays
        rewardsAll = Variable(torch.FloatTensor(rewardsAll))
        doneAll = Variable(torch.FloatTensor(doneAll))
        # Compute Q values
        qValuesCurr = modelOpt.forward(Variable(torch.from_numpy(np.concatenate(currObsAll, axis=0)).float()))
        qValuesNext = modelFix.forward(Variable(torch.from_numpy(np.concatenate(nextObsAll, axis=0)).float()))
        # Compute prediction
        pred = qValuesCurr[range(timeSteps), actionsAll]
        # Compute target
        target = rewardsAll + discount * ((1 - doneAll) * qValuesNext.max(dim=1)[0])
        # Accumulate loss, unwrapping target vector
        loss = loss_func(pred, Variable(target.data))
        # Backward propogation
        loss.backward()
        # Update parameters
        optimizer.step()
        # Compute averaged episode loss
        episodeLoss = loss.data[0] / timeSteps
        # Print timesteps, score
        print EPISODE_STRING.format(ep, timeSteps+1, int(score), time.time()-startTime, episodeLoss)
        # Store score and loss
        scores[ep] = score
        losses[ep] = episodeLoss

# Function to train with N-step Q-Learning
def trainNstepQL(modelOpt, modelFix, scores, losses, episode):
    # Create separate environment
    env = gym.make(gameString)
    # For each episode...
    while episode.value < numEpisodes:
        # Get current episode and increment tracked episode
        with episode.get_lock():
            ep = episode.value
            episode.value += 1
        # Calculate epsilon
        epsilon = 1.0 - (1.0 - EPSILON_MIN) / EPSILON_STOP * (float(ep) / numEpisodes)
        epsilon = max(epsilon, EPSILON_MIN)
        # Initialize time-steps, score, and loss
        timeSteps = 0
        score = 0.0
        # Reset environment and set done to false
        currObsStacked = np.tile(processObs(env.reset(), isRAM), tileDims)
        lives = env.env.ale.lives()
        done = False
        # Render
        env.render(close=closeRender)
        # Track observations, actions, rewards, and terminal
        currObsAll = []
        actionsAll = []
        rewardsAll = []
        # Reset gradient
        optimizer.zero_grad()
        # While game is not done...
        while not done:
            # If should choose random action...
            if random.random() < epsilon:
                # Choose random action
                action = env.action_space.sample()
            # Else...
            else:
                # Compute Q values using thread model
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
            # Render
            env.render(close=closeRender)
            # Add to lists
            currObsAll.append(np.expand_dims(currObsStacked, 0))
            actionsAll.append(action)
            rewardsAll.append(reward)
            # Update current observation, score, time
            currObsStacked = np.vstack((currObsStacked[1:], np.expand_dims(processObs(nextObs, isRAM), 0)))
            score += reward
            timeSteps += 1
        # Turn to numpy arrays
        rewardsAll = Variable(torch.FloatTensor(rewardsAll))
        # Compute Q values
        qValuesCurr = modelOpt.forward(Variable(torch.from_numpy(np.concatenate(currObsAll, axis=0)).float()))
        # Initialize accumulated reward and loss to zero
        R = Variable(torch.zeros(1), requires_grad=False)
        loss = 0
        # For each timestep, backwards...
        for t in range(timeSteps)[::-1]:
            # Accumulate reward and loss
            R = R * discount + rewardsAll[t]
            loss += loss_func(qValuesCurr[t, actionsAll[t]], R)
        # Backward propogation
        loss.backward()
        # Update parameters
        optimizer.step()
        # Compute averaged episode loss
        episodeLoss = loss.data[0] / timeSteps
        # Print timesteps, score
        print EPISODE_STRING.format(ep, timeSteps+1, int(score), time.time()-startTime, episodeLoss)
        # Store score and loss
        scores[ep] = score
        losses[ep] = episodeLoss
        # If we have performed 25 episodes...
        if ep % 25 == 0:
            # Update fixed model
            modelFix.load_state_dict(modelOpt.state_dict())


#################
### MAIN CODE ###
#################

# Global parameters
MEMORY_CAPACITY = 50000
BATCH_SIZE = 32
NUM_FRAMES = 4
CONVOLUTION_BRANCH = 2
MODEL_UPDATE = 25
NUM_PROCESSES = 4
EPSILON_MIN = 0.1
EPSILON_STOP = 0.7
GRAY_WEIGHTS = [0.3, 0.6, 0.1]
MOVE_AVG = 100
EPISODE_STRING = "Ep {0:>6}: steps {1:>6}, score {2:>6}, time {3:>9.2f}, loss {4:>10.2f}"

TRAINING_FUN = {"DQN_EP": trainDQNep,
                "DQN_AS": trainDQNasync,
                "NSTEP_QL": trainNstepQL}

# If main...
if __name__ == "__main__":
    # Set parameters
    game = "Breakout"
    isRAM = True
    numEpisodes = 12000
    closeRender = True
    isMultiprocess = True
    oneLife = False
    discount = 0.99
    trainType = "DQN_AS"
    loss_func = torch.nn.SmoothL1Loss()
    # Create game-string
    gameString = game + isRAM * "-ram" + "-v4"
    # Initialize test environment to get information
    testEnv = gym.make(gameString)
    obsDims = processObs(testEnv.reset(), isRAM).shape
    tileDims = tuple([NUM_FRAMES] + len(obsDims) * [1])
    # Specify model arguments and initialize models
    modelType = NNRAM if isRAM else NNscreen
    modelArgs = {"num_frames": NUM_FRAMES,
                 "b": CONVOLUTION_BRANCH,
                 "input_dim": obsDims,
                 "output_dim": testEnv.action_space.n}
    modelOpt = modelType(**modelArgs)
    modelFix = modelType(**modelArgs)
    modelFix.load_state_dict(modelOpt.state_dict())
    # If training a traditional DQN...
    if trainType == "DQN_EP":
        # Change number of processes to 1 so memory is used
        NUM_PROCESSES = 1
    # Have models share memory
    modelOpt.share_memory()
    modelFix.share_memory()
    # Initialize optimizers
    optimizer = torch.optim.Adam(modelOpt.parameters())
    # Initialize shared memory information
    scores = mp.Array("d", [0] * numEpisodes)
    losses = mp.Array("d", [0] * numEpisodes)
    episode = mp.Value("i", 0)
    # Get number of actual processes
    numProcesses = NUM_PROCESSES if isMultiprocess else 1
    # Start the training timer
    startTime = time.time()
    # Multiprocess
    processes = []
    for rank in range(numProcesses):
        p = mp.Process(target=TRAINING_FUN[trainType], args=(modelOpt, modelFix, scores, losses, episode))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # Smooth scores
    # scores = scores[:]
    # smoothScores = np.zeros((len(scores)-MOVE_AVG))
    # for i in range(len(smoothScores)):
    #     smoothScores[i] = np.average(scores[i:i+100])
    # Plot smoothed scores
    plt.plot(scores[:])
    plt.savefig("plots/" + game + isRAM * "-ram" + "_" + str(numEpisodes) + "e_scores.png")
    plt.close()
    with open("plots/" + game + isRAM * "-ram" + "_" + str(numEpisodes) + "e_scores_" + trainType + ".pk", "wb") as f:
        pickle.dump(scores[:], f)
    # Plot loss and save
    plt.plot(losses[:])
    plt.savefig("plots/" + game + isRAM * "-ram" + "_" + str(numEpisodes) + "e_losses.png")
    plt.close()
    # Save model
    with open("models/" + isRAM * "RAM/" + (not isRAM) * "screen/" + game, "wb") as f:
        torch.save(modelOpt, f)