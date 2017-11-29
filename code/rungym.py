import gym
import time
from agents import *

# Set game
game = "MsPacman"
# Create environment
env = gym.make(game + "-ram-v0")
# Create agent
# agent = RandomAgent()
# agent = BFSAgent(maxActions=1, oneAction=False)
agent = DQNRAMagent(game + "-ram")
# Set maximum number of plan-ahead actions and number of episodes
numEpisodes = 100
closeRender = False
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
        # Get actions
        actions = agent.getActions(env, obs)
        # For each action in the max path...
        for action in actions:
            # Take the action
            obs, reward, done, _ = env.step(action)
            # Render
            env.render(close=closeRender)
            # Add to score
            score += reward
            # If actually done...
            if done:
                # Terminate
                break
        # Increment time
        timeSteps += len(actions)
    # Print timesteps, score
    print "Episode {0:>3}: {1:>4} steps, score {2:>6}, time {3:>6.2f}".format(episode, timeSteps+1, int(score), time.time()-startTime)
    # Store score
    scores.append(score)
# Print average score
print "Average score is {}".format(float(sum(scores)) / numEpisodes)