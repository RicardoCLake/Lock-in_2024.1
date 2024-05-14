# Author: Lucas José Velôso de Souza <lucasjose.velosodesouza@student-cs.fr>

from torch import nn
from torch import Tensor

class Flatten(nn.Module):

    def __init__(self, num_neurons_flatten: int):
      
        """
        
        Constructor for the Flatten module.
        
        Module that flattens the input tensor into a 2D tensor.

        Args:
        - num_neurons_flatten (int): number of neurons to flatten the input tensor to

        Attributes:
        - num_neurons_flatten (int): number of neurons to flatten the input tensor to
        
        """
        super(Flatten, self).__init__()
        self.num_neurons_flatten = num_neurons_flatten

    def forward(self, x: Tensor) -> Tensor:
      
        """
        
        Forward pass of the Flatten module.

        Args:
        - x (Tensor): input tensor

        Returns:
        - out (Tensor): flattened tensor
        
        """
        out = x.contiguous().view(-1, self.num_neurons_flatten)
        return out
