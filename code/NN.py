import torch
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.lin = torch.nn.Linear(input_dim*num_frames*(b**2), output_dim)
    # Forward function
    def forward(self, obs):
        x = F.dropout(F.relu(self.conv1(obs)))
        x = F.dropout(F.relu(self.conv2(x)))
        x = self.lin(x.view(x.size(0),-1))
        return x