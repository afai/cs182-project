# # import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# # from NN import *

# v = torch.ones(4, 4)
# print v.view(v.size(0), -1)


# # with open("DQNmodels/Atlantis-ram", "rb") as f:
# #     model = torch.load(f)

# # # Set game
# # game = "Atlantis"
# # # Create environment
# # env = gym.make(game + "-ram-v0")
# # # Set maximum number of plan-ahead actions and number of episodes
# # numEpisodes = 100
# # closeRender = True
# # # Initialize time-steps and score to zero
# # timeSteps = 0
# # score = 0
# # # Reset environment and set done to false
# # obs = env.reset()
# # done = False
# # # Render
# # env.render(close=closeRender)
# # # While game is not done...
# # while not done:
# #     # Get actions
# #     actions = agent.getActions(env, obs)
# #     # For each action in the max path...
# #     for action in actions:
# #         # Take the action
# #         obs, reward, done, _ = env.step(action)
# #         # Render
# #         env.render(close=closeRender)
# #         # Add to score
# #         score += reward
# #         # If actually done...
# #         if done:
# #             # Terminate
# #             break
# #     # Increment time
# #     timeSteps += len(actions)
# # # Print timesteps, score
# # print "Episode {0:>3}: {1:>4} steps, score {2:>6}".format(episode, timeSteps+1, int(score))