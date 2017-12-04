import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

KERNEL1, STRIDE1 = ((8, 8), 4)
KERNEL2, STRIDE2 = ((4, 4), 2)
KERNEL3, STRIDE3 = ((3, 3), 1)
HIDDEN_DIM = 256

# Function for computing the dimensions and padding for convolution layers
def computeConvolutionDims(dims, kernel_size, stride):
    nextdims = tuple([int(math.ceil((dims[i] - kernel_size[i]) / float(stride))) + 1 for i in range(len(dims))])
    paddings = tuple([((nextdims[i] - 1) * stride + kernel_size[i]) - dims[i] for i in range(len(dims))])
    return nextdims, paddings

# Class for neural network for RAM
class NNRAM(torch.nn.Module):
    # Initialize
    def __init__(self, num_frames, b, input_dim, output_dim):
        super(NNRAM, self).__init__()
        # Store number of frames
        self.num_frames = num_frames
        # Convolution 1
        self.conv1 = torch.nn.Conv1d(num_frames, num_frames*b, kernel_size=1)
        # Convolution 2
        self.conv2 = torch.nn.Conv1d(num_frames*b, num_frames*(b**2), kernel_size=1)
        # Linear 1
        self.lin = torch.nn.Linear(input_dim[0]*num_frames*(b**2), output_dim)
    # Forward function
    def forward(self, obs):
        x = F.dropout(F.relu(self.conv1(obs)))
        x = F.dropout(F.relu(self.conv2(x)))
        x = self.lin(x.view(x.size(0),-1))
        return x

# Class for neural network for screen
class NNscreen(torch.nn.Module):
    # Initialize
    def __init__(self, num_frames, b, input_dim, output_dim):
        super(NNscreen, self).__init__()
        # Store number of frames
        self.num_frames = num_frames
        # Compute next dimensions and necessary padding
        nextdims1, pad1 = computeConvolutionDims(input_dim, KERNEL1, STRIDE1)
        # Convolution 1
        self.conv1 = torch.nn.Conv2d(num_frames*(b**0), num_frames*(b**1), kernel_size=KERNEL1, stride=STRIDE1, padding=pad1)
        # Compute next dimensions and necessary padding
        nextdims2, pad2 = computeConvolutionDims(nextdims1, KERNEL2, STRIDE2)
        # Convolution 2
        self.conv2 = torch.nn.Conv2d(num_frames*(b**1), num_frames*(b**2), kernel_size=KERNEL2, stride=STRIDE2, padding=pad2)
        # Compute next dimensions and necessary padding
        nextdims3, pad3 = computeConvolutionDims(nextdims2, KERNEL3, STRIDE3)
        # Convolution 3
        self.conv3 = torch.nn.Conv2d(num_frames*(b**2), num_frames*(b**3), kernel_size=KERNEL3, stride=STRIDE3, padding=pad3)
        # Linear 1
        self.lin1 = torch.nn.Linear(np.prod(list(nextdims3))*num_frames*(b**3), HIDDEN_DIM)
        # Linear 2
        self.lin2 = torch.nn.Linear(HIDDEN_DIM, output_dim)
    # Forward function
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.lin1(x.view(x.size(0),-1)))
        x = self.lin2(x.view(x.size(0),-1))
        return x