# cs182-project
CS 182 Final Project

Training:
In train.py, specify game, whether or nor RAM is the input, and which training algorithm to use. The model will be saved in modelswrite, scores will be serialized in the plots folder, and losses and scores will be plotted (unsmoothly) in the plots folder as well.

Testing:
Make sure to copy model from modelswrite to modelsread. This is to prevent accidental overwriting of current models. In rungym.py, specify game, whether or nor RAM is the input, and which training algorithm to use; the model will be read from modelsread.

Other Files:
NN.py - contains the convolutional neural networks, used by both train.py
ExperienceReplay.py - contains the experience replay class, used by train.py
agents.py - contains the agents themselves, used by rungym.py

Parameter warnings:
In train.py, set NUM_PROCESSES to the number of cores you want to use. If running on screen inputs, make sure to set MEMORY_CAPACITY (DQN with experience replay) and TIMESTEP_LIMIT (Asynchronous N-Step Q-Learning) appropriately according to your computational resources. Using 4 cores and 8 GB of memory, I used a TIMESTEP_LIMIT of 800.