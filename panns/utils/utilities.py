"""This module provides methods necessary to configure features and labels
Methods available:
    one hot encode
    Change data type from 32-bit float to 16-bit int and applies scaling between -32767 and 32767
    Change data type from 16-bit int to 32-bit float and applies normalization
    Change data type from numpy to torch
    Mixup operation
    Update a dictionary with a specific key, appending the value to a list
    Set integer labels for each class from an audio directory with the following structure
        bird_species2

        bird_species2
        ...

        bird_speciesn
    negative log likelihood
"""
import os
import numpy as np
import torch

def to_one_hot(k, classes_num):
    """One hot encode for a given k integer and class_number number of classes"""
    target = np.zeros(classes_num)
    target[k] = 1
    return target

def float32_to_int16(time_series):
    """To scale audio between -32767 and 32767"""
    # assert np.max(np.abs(x)) <= 1.
    if np.max(np.abs(time_series)) > 1.:
        time_series /= np.max(np.abs(time_series))
    return (time_series * 32767.).astype(np.int16)

def int16_to_float32(time_series):
    """To normalize audio between -1 and 1 as 32-bit floating point"""
    return (time_series / 32767.).astype(np.float32)

# pylint: disable=invalid-name
def move_data_to_device(x, device):
    """Convert numpy data to torch"""
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

# pylint: disable=redefined-builtin
def append_to_dict(dict, key, value):
    """Update or create a dictionary with 'key' appending the value to a list"""
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def set_labels(dataset_dir):
    """Find the classes of a directory and returns them with a
    dictionary{label: integer} for each class from an audio directory """
    labels = []
    for paths in os.walk(dataset_dir):
        if len(paths[1]) != 0:
            labels.extend(paths[1])

    labels = sorted(labels)
    lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
    return labels, lb_to_idx

# pylint: disable=missing-function-docstring
def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss
