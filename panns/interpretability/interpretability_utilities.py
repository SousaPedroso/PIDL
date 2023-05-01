# pylint: disable=[missing-module-docstring, wrong-import-position]
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# pylint: disable=import-error
from utils.utilities import int16_to_float32, move_data_to_device, set_labels
import numpy as np
import h5py
import matplotlib.pyplot as plt

def plot_frame_attributions(attributions, title="Average Frames importance using Deconvolution"):
    """ Auxiliar method to plot the attributions"""
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(attributions[0])), np.mean(attributions.cpu().detach().numpy(), axis=0))
    plt.xlabel("Frames")
    plt.title(title)

def load_workspace_file(workspace_file_path, ref_fold, dataset_dir, device):
    """ Load the data from the workspace for a given fold and returns the waveform and labels
        with the respectives in.
        Also returns the labels and label setter of the dataset.
    """
    with h5py.File(workspace_file_path, 'r') as h5_file:
        # pylint: disable=no-member
        folds = h5_file['fold'][:].astype(np.float32)

        indexes = np.where(folds == int(ref_fold))[0]
        inp_data = int16_to_float32(h5_file["waveform"][indexes])
        inp_data = move_data_to_device(inp_data, device)
        labels = h5_file["target"][indexes]
        labels = move_data_to_device(labels, device)

    dataset_labels, lb_to_idx = set_labels(dataset_dir)
    return inp_data, indexes, labels, dataset_labels, lb_to_idx


def zero_crossing_rate(frame):
    """ Compute zero crossing rate for a frame.
        https://github.com/tyiannak/pyAudioAnalysis
    """
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)
