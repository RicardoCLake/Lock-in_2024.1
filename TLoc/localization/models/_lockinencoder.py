# Author: Lucas José Velôso de Souza <lucasjose.velosodesouza@student-cs.fr>

from torch import nn
from ._positionalencoding import PositionalEncoding
from torch import Tensor,IntTensor,BoolTensor

class LockinEncoder(nn.Module):
  
  def __init__(self, num_embeddings: int, embedding_dim: int, nheads: int, dim_feedforward: int,
               transformer_activation: nn.Module, num_transformer_blocks: int, dropout: float = 0.1):
    
    """

    LockinEncoder module consisting of an embedding layer, positional encoder,
    and a transformer encoder with multiple layers.
    
    A module that encodes input sequences using a stack of Transformer Encoder blocks.

    Args:

    - num_embeddings (int): the size of the dictionary of embeddings.
    - embedding_dim (int): the dimension of each embedding vector. Must be an integer divisible by nheads.
    - nheads (int): the number of attention heads in each Transformer block.
    - dim_feedforward (int): the size of the feedforward layer in each Transformer block.
    - transformer_activation (nn.Module): the activation function used in the feedforward layer of each Transformer block.
    - num_transformer_blocks (int): the number of Transformer blocks to stack.
    - dropout (float): the dropout probability to use throughout the module.

    Attributes:
    
    - embedding (nn.Embedding): the embedding layer used to convert input tokens to embedding vectors.
    - pos_encoder (PositionalEncoding): the positional encoding layer used to add positional information to the embeddings.
    - transformer_block (nn.TransformerEncoderLayer): a single Transformer Encoder block.
    - transformer (nn.TransformerEncoder): the stack of Transformer Encoder blocks.
        

    """
    
    if embedding_dim % nheads != 0 :
      raise ValueError("embedding_dim must be divisible by nheads")
      
    super(LockinEncoder,self).__init__()

    # Initialize embedding layer
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    # Initialize positional encoder
    self.pos_encoder = PositionalEncoding(embedding_dim,embedding_dim,dropout)

    # Initialize transformer block
    self.transformer_block = nn.TransformerEncoderLayer(embedding_dim, nheads, dim_feedforward, dropout,
                                                        transformer_activation(), batch_first=True)

    # Initialize transformer with stacked blocks
    self.transformer = nn.TransformerEncoder(self.transformer_block, num_transformer_blocks)

  def forward(self, routers: IntTensor, routers_attn_mask: BoolTensor,
              routers_key_padding_mask: BoolTensor) -> Tensor:
    """

    Forward pass of the LockinEncoder module.

    Args:
      routers (IntTensor): Input tensor of shape (batch_size, seq_len) representing
                           the routers to be encoded.
      routers_attn_mask (BoolTensor): Input tensor of shape (batch_size, nheads, seq_len, seq_len)
                                      representing the attention mask to be applied on the routers.
      routers_key_padding_mask (BoolTensor): Input tensor of shape (batch_size, seq_len)
                                             representing the key padding mask to be applied on the routers.

    Returns:
      A tensor representing the encoded routers.

    """
    batch_size, nheads, seq_len, _ = routers_attn_mask.size()
    routers_attn_mask = routers_attn_mask.view(batch_size * nheads, seq_len, seq_len)

    # Embed the routers
    embed_routers = self.embedding(routers)

    # Apply positional encoding on the embedded routers
    pos_embed_routers = self.pos_encoder(embed_routers)

    # Apply the transformer on the embedded and encoded routers
    encoded_routers = self.transformer(pos_embed_routers, routers_attn_mask, routers_key_padding_mask)

    return encoded_routers
