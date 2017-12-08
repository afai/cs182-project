import gym
import time
import numpy as np
from agents import *

EPISODE_STRING = "Ep {0:>6}: steps {1:>6}, score {2:>6}, time {3:>9.2f}"

# Set parameters to change
game = "Breakout"
isRAM = False
trainType = "NSTEP_QL" # Can be "DQN_EP", "DQN_AS", or "NSTEP_QL"
# Parameters not to change
numEpisodes = 100
closeRender = True
# Create environment and agent
env = gym.make(game + isRAM * "-ram" + "-v4")
# agent = RandomAgent()
agent = DQNagent(game, isRAM=isRAM, trainType=trainType, env=env, epsilon=0.05) # Use epsilon 0.05 for Breakout, 0.00 for others
# Store episode scores
scores = []
# For each episode...
for episode in range(numEpisodes):
    # Start the timer
    startTime = time.time()
    # Initialize time-steps and score to zero
    timeSteps = 0
    score = 0
    # Reset environment and set done to false
    obs = env.reset()
    done = False
    # Render
    env.render(close=closeRender)
    # While game is not done...
    while not done:
        # Get action
        action = agent.getAction(env, obs)
        # Take the action
        obs, reward, done, _ = env.step(action)
        # Render
        env.render(close=closeRender)
        # Add to score
        score += reward
        # Increment time
        timeSteps += 1
    # Print timesteps, score
    print EPISODE_STRING.format(episode, timeSteps+1, int(score), time.time()-startTime)
    # Store score
    scores.append(score)
# Print average score
print "Average score is {}".format(np.mean(scores))
print "Standard deviation is {}".format(np.std(scores))