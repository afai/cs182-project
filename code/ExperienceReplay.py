import numpy as np

# Class for remembering transitions
class ExperienceReplay():
    # Initialize
    def __init__(self, capacity, batchSize, numFrames, obsDims):
        # Remember capacity and batch size
        self.capacity = capacity
        self.batchSize = batchSize
        # Compute dimensions for transition states in memory
        transSTSdims = [capacity, numFrames] + [d * (1 + (not i)) for i, d in enumerate(obsDims)]
        # Each row in memory has current state, next state, action, reward, and done
        self.transSTS = np.empty(transSTSdims, dtype="uint8")
        self.transARD = np.empty((capacity, 3), dtype="uint8")
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