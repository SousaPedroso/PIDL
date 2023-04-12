"""This file provides methods necessary to configure features and labels
Methods available:
    one hot encode
    Change data type from 32-bit float to 16-bit int and applies scaling between -32767 and 32767
    Change data type from 16-bit int to 32-bit float and applies normalization
    Change data type from numpy to torch
"""
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
    