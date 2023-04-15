"""Auxiliary classes to train and evaluate PANNs models
https://github.com/qiuqiangkong/panns_transfer_to_gtzan/blob/master/utils/data_generator.py
"""
# pylint: disable=import-error
from utils.utilities import int16_to_float32
import numpy as np
import h5py

class GtzanDataset:
    """This class takes the meta of an audio clip as input, and return 
    the waveform and target of the audio clip. This class is used by DataLoader. 
    """
    def __init__(self):
        pass

    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'wav_path': str, 
            'index_in_wav': int}
        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(hdf5_path, 'r') as h5_file:
            audio_name = h5_file['audio_name'][index_in_hdf5].decode()
            waveform = int16_to_float32(h5_file['waveform'][index_in_hdf5])
            target = h5_file['target'][index_in_hdf5].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target
        }

        return data_dict

class TrainSampler:
    """Balanced sampler. Generate batch meta for training.
    
    Args:
        hdf5_path: string
        holdout_fold: string
        batch_size: int
        random_seed: int
    """
    def __init__(self, hdf5_path, holdout_fold, batch_size, random_seed=135):

        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.random_state = np.random.default_rng(random_seed)

        with h5py.File(hdf5_path, 'r') as h5_file:
            # pylint: disable=no-member
            self.folds = h5_file['fold'][:].astype(np.float32)

        self.indexes = np.where(self.folds != int(holdout_fold))[0]
        self.audios_num = len(self.indexes)

        # Shuffle indexes
        self.random_state.shuffle(self.indexes)

        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav',
             'hdf5_path': 'xx/balanced_train.wav',
             'index_in_wav': 15734,
             'target': [0, 1, 0, 0, ...]},
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)

                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': index
                })
                i += 1

            yield batch_meta

    # pylint: disable=missing-function-docstring
    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer
        }
        return state

    # pylint: disable=missing-function-docstring
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']

class EvaluateSampler:
    """Balanced sampler. Generate batch meta for validation.
    
    Args:
        hdf5_path: string
        batch_size: int
        black_list_csv: string
        random_seed: int
    """
    def __init__(self, hdf5_path, holdout_fold, batch_size):

        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

        with h5py.File(hdf5_path, 'r') as h5_file:
            # pylint: disable=no-member
            self.folds = h5_file['fold'][:].astype(np.float32)

        self.indexes = np.where(self.folds == int(holdout_fold))[0]
        self.audios_num = len(self.indexes)

    def __iter__(self):
        """Generate batch meta for evaluate. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'wav_path': 'xx/balanced_train.h5', 
             'index_in_wav': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for i in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': self.indexes[i]
                })

            pointer += batch_size
            yield batch_meta

def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict
