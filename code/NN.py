import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Class for neural network for RAM
class NNRAM(torch.nn.Module):
    # Initialize
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNRAM, self).__init__()
        # Linear 1
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        # Linear 2
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
    # Forward function
    def forward(self, obs):
        return self.lin2(F.relu(self.lin1(obs)))