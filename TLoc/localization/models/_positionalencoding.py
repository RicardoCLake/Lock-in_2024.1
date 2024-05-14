# Author: Lucas José Velôso de Souza <lucasjose.velosodesouza@student-cs.fr>

import math
import torch
from torch import nn
from torch import Tensor

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
    
        """

        Constructor for the PositionalEncoding class.
        Based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Args:
            -d_model (int): The number of expected features in the input.
            -dropout (float): The probability of an element to be zeroed in the dropout layer.
            -max_len (int): The maximum length of the incoming sequence.

        Attributes:
            -dropout (nn.Dropout): The dropout layer applied to the positional encoding.
            -pe (Tensor): The positional encoding tensor with shape (max_len, 1, d_model), 
                         generated using sine and cosine functions based on position and 
                         exponential decay of frequency.
        """
            
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Generate position indices
        position = torch.arange(max_len).unsqueeze(1)
        
        # Calculate the div_term
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Generate the positional encoding values
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
    
        """
        
        Applies positional encoding to a sequence of embeddings.
        
        Args:
            x: Tensor - A tensor of shape [batch_size, seq_len, embedding_dim].
            
        Returns:
            Tensor - A tensor of shape [batch_size, seq_len, embedding_dim] with positional encoding applied.
            
        """
        
        # Transpose to [seq_len, batch_size, embedding_dim]
        x = x.transpose(0,1)
        
        # Add positional encoding to the embeddings
        x = x + self.pe[:x.size(0)]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transpose back to [batch_size, seq_len, embedding_dim]
        x = x.transpose(1,0)
        return x
