# Author: Lucas José Velôso de Souza <lucasjose.velosodesouza@student-cs.fr>

import torch
import numpy as np
from typing import Tuple

class Normalizer:

  def __init__(self, src_dim: int, nheads: int, train_ids: list, padding_idx: int = 0, mask_idx: int = 1):

    """

    Initializes the Normalizer object.
    
    The Normalizer class is a utility class designed to preprocess input data for a BERT language model.
    Specifically, the class implements a preprocessing method for applying prediction masks and padding tokens to
    network router token sequences. The resulting output is a set of input tensors for the BERT model, including
    network router tensors, attention masks, and padding masks. 

    Args:
    - src_dim (int): the source dimension, i.e. the maximum length of input sequences after padding.
    - nheads (int): the number of attention heads in the neural network model.
    - padding_idx (int): the index value used for padding input sequences.
    - mask_idx (int): the index value used for masking out a random element in each input sequence.
    - train_ids (list): a list for storing training IDs, which are used for replacing masked elements in the input sequences during pre-training.


    Attributes:
    - src_dim (int): the source dimension, i.e. the maximum length of input sequences after padding.
    - nheads (int): the number of attention heads in the neural network model.
    - padding_idx (int): the index value used for padding input sequences.
    - mask_idx (int): the index value used for masking out a random element in each input sequence.
    - train_ids (list): a list for storing training IDs, which are used for replacing masked elements in the input sequences during pre-training.

    """
    
    # Set the source dimension attribute to the given argument value.
    self.src_dim = src_dim
    
    # Set the number of attention heads attribute to the given argument value.
    self.nheads = nheads
    
    # Set the padding index attribute to the given argument value. Default is 0.
    self.padding_idx = padding_idx
    
    # Set the mask index attribute to the given argument value. Default is 1.
    self.mask_idx = mask_idx
    
    # Create an empty list for storing training IDs.
    self.train_ids = train_ids

  def selecting_target(self, phrase: list) -> Tuple[list,int]:

    """

    Selects a target element in the input phrase and masks it out, replaces it with a random element, or leaves it unchanged
    based on a uniform probability distribution.

    Args:
    - phrase (list): the input phrase to process.

    Returns:
    - A tuple containing the processed phrase and the selected target element.

    """
    
    # Generate a random probability between 0 and 1.
    p = np.random.uniform(0,1)
    
    # Select a random index in the phrase between 0 and the minimum of the phrase length and the source dimension.
    target_index = np.random.randint(0, min(len(phrase), self.src_dim))
    
    # Select the target element at the target index.
    target = phrase[target_index]
    
    # With probability 0.8, mask out the target element by replacing it with the mask index value.
    if p <= 0.8:
        phrase[target_index] = self.mask_idx
        
    # With probability 0.1, replace the target element with a random element from the training IDs list.
    elif p <= 0.9:
        phrase[target_index] = np.random.choice(self.train_ids)
    
    # With probability 0.1, leave the target element unchanged.
    else:
        pass
    
    # Return the processed phrase and the selected target element as a tuple.
    return phrase, target

  def pre_padding(self, phrase: list) -> list:

    """

    Pads an input phrase with the padding index value up to the source dimension size, or truncates it to the source
    dimension size if it exceeds it.

    Args:
    - phrase (list): the input phrase to pad.

    Returns:
    - The padded phrase.

    """
    
    # If the length of the phrase is less than the source dimension size, pad it with the padding index value
    if len(phrase) < self.src_dim:
        while len(phrase) < self.src_dim:
            phrase.append(self.padding_idx)
            
    # If the length of the phrase exceeds the source dimension size, truncate it to the source dimension size.
    else:
        phrase = phrase[:self.src_dim]
        
    # Return the padded phrase.
    return phrase
  
  def pretraining_mask(self, phrase: list) -> Tuple[list, int]:

    """

    Applies a pretraining mask to an input phrase by replacing a target element with the mask index or a random element
    from the training data. Then, pads the phrase with the padding index value up to the source dimension size or truncates
    it to the source dimension size if it exceeds it.

    Args:
    - phrase (list): the input phrase to apply the pretraining mask to.

    Returns:
    - The phrase with the pretraining mask applied.
    - The index of the target element in the original phrase.

    """
    
    # Select a target element to replace with the mask index or a random element from the training data.
    phrase, target = self.selecting_target(phrase)
    
    # Pad the phrase with the padding index value up to the source dimension size or truncate it to the source dimension
    # size if it exceeds it.
    phrase = self.pre_padding(phrase)
    
    # Return the phrase with the pretraining mask applied and the index of the target element in the original phrase.
    return phrase, target

  def preprocess_pretraining_mask(self, X: list) -> Tuple[torch.IntTensor, torch.BoolTensor, torch.BoolTensor, torch.IntTensor]:

    """

    Preprocesses a list of input phrases by applying a pretraining mask to each phrase, padding each phrase with the padding
    index value up to the source dimension size or truncating it to the source dimension size if it exceeds it, creating
    attention masks and key padding masks for each phrase, and returning the processed data as tensors.

    Args:
    - X (list): the list of input phrases to preprocess.

    Returns:
    - A tensor of the preprocessed phrases with the pretraining mask applied.
    - A tensor of the attention masks for each phrase.
    - A tensor of the key padding masks for each phrase.
    - A tensor of the target elements for each phrase.

    """
    
    # Create empty lists to store the preprocessed phrases and target elements.
    routers = []
    targets = []

    # For each input phrase in the list of phrases to preprocess:
    for phrase in X:
        
        # Apply a pretraining mask to the phrase.
        phrase, target = self.pretraining_mask(phrase)
        
        # Pad the phrase with the padding index value up to the source dimension size or truncate it to the source dimension
        # size if it exceeds it.
        phrase = self.pre_padding(phrase)
        
        # Add the preprocessed phrase and target element to the routers and targets lists.
        routers.append(phrase)
        targets.append(target)

    # Convert the key padding masks for each phrase to a tensor.
    key_padding_mask = [np.where(np.array(phrase)==self.padding_idx,1,0) for phrase in routers]
    key_padding_mask = np.array(key_padding_mask)
    key_padding_mask = key_padding_mask.astype(np.float32)
    key_padding_mask = torch.BoolTensor(key_padding_mask)

    # Create attention masks for each phrase.
    attention_masks = [[[mask.numpy() for _ in range(self.src_dim)] for _ in range(self.nheads)] for mask in key_padding_mask]

    # Convert the routers and targets lists to tensors.
    routers = np.array(routers)
    routers = routers.astype(np.int32)
    routers = torch.IntTensor(routers)

    targets = np.array(targets)
    targets = targets.astype(np.int32)
    targets = torch.IntTensor(targets)

    attention_masks = np.array(attention_masks)
    attention_masks = attention_masks.astype(np.float32)
    attention_masks = torch.BoolTensor(attention_masks)

    # Return the preprocessed data as tensors.
    return routers, attention_masks, key_padding_mask, targets


    # Convert the routers and targets lists to tensors.
    routers = np.array(routers)
    routers = routers.astype(np.int32)
    routers = torch.IntTensor(routers)

    targets = np.array(targets)
    targets = targets.astype(np.int32)
    targets = torch.IntTensor(targets)

    attention_masks = np.array(attention_masks)
    attention_masks = attention_masks.astype(np.float32)
    attention_masks = torch.BoolTensor(attention_masks)

    # Return the preprocessed data as tensors.
    return routers, attention_masks, key_padding_mask, targets
