# pylint: disable=[missing-module-docstring, wrong-import-position]
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# pylint: disable=import-error
from math import floor
from utils.utilities import int16_to_float32, move_data_to_device, set_labels
import numpy as np
import librosa
import h5py
import matplotlib.pyplot as plt

def plot_frame_attributions(attributions, title="", sample_rate=32000):
    """ Auxiliar method to plot the attributions"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(attributions[0])), np.mean(attributions.cpu().detach().numpy(), axis=0))
    plt.xticks(np.arange(0, attributions.size()[1], int(sample_rate/10)),
        np.arange(0, round(len(attributions[0])/sample_rate, 2), 0.10, dtype=np.float32))
    if not title.isspace():
        plt.title(title)

# to-do refactor this method, below and above in one
# pylint: disable=line-too-long
def plot_layer_attribution_importance(attributions, title=""):
    """ Auxiliar method to plot the attributions mean for each input """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(attributions.size()[0]), np.mean(attributions.cpu().detach().numpy(), axis=(1,2,3)))
    plt.xlabel("Entrada")
    plt.ylabel("Atribuição")
    if not title.isspace():
        plt.title(title)

def plot_audio_attributions(attributions, title=""):
    """ Auxiliar method to plot the attributions"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(attributions.size()[0]), np.mean(attributions.cpu().detach().numpy(), axis=1))
    plt.xlabel("Entrada")
    plt.ylabel("Atribuição")
    if not title.isspace():
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
        audios_name = move_data_to_device(h5_file["audio_name"][indexes], device)

    dataset_labels, lb_to_idx = set_labels(dataset_dir)
    return inp_data, indexes, labels, dataset_labels, lb_to_idx, audios_name


def zero_crossing_rate(frame):
    """ Compute zero crossing rate for a frame.
        https://github.com/tyiannak/pyAudioAnalysis
    """
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)

# pylint: disable=[missing-function-docstring, too-many-locals]
def evaluate_inputs(model, audios_path, audio_class, sample_rate, duration):
    rng = np.random.default_rng(135)

    effective_length = floor(sample_rate * duration)

    full_path = os.path.join(audios_path, audio_class)

    audios = []
    audios_name = []
    for file in os.listdir(full_path):
        offset = rng.integers(
            max(librosa.get_duration(filename=os.path.join(full_path, file))-duration, 1)
        )

        (audio, _) = librosa.core.load(os.path.join(full_path, file),
                offset=offset, sr=None, duration=duration)
        # get only one channel if stereo audio
        if len(audio.shape) == 2:
            audio = audio[:, 0]

        len_audio = len(audio)
        if len_audio < effective_length:
            new_audio = np.zeros(effective_length, dtype=audio.dtype)
            start = rng.integers(effective_length - len_audio)
            new_audio[start:start + len_audio] = audio
            audio = new_audio.astype(np.float32)
        elif len_audio > effective_length:
            start = rng.integers(len_audio - effective_length)
            audio = audio[start:start + effective_length].astype(np.float32)
        else:
            audio = audio.astype(np.float32)

        audios.append(audio)
        audios_name.append([os.path.join(full_path, file), file])

    audios_evaluation = model(move_data_to_device(np.array(audios), "cpu"))
    return audios_evaluation, audios_name
