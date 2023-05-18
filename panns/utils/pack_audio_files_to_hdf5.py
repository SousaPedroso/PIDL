"""Get an audio dataset and defines an HDF5 file from it to train the PANNs
See https://github.com/qiuqiangkong/panns_transfer_to_gtzan/blob/master/utils/features.py
"""
import os
from math import floor
import argparse
import numpy as np
import h5py
import librosa
from utilities import to_one_hot, float32_to_int16, set_labels

# pylint: disable=missing-function-docstring
def pack_audio_files_to_hdfs(args: argparse.Namespace):
    dataset_dir = args.dataset_dir
    workspace_dir = args.workspace_dir
    suffix = args.suffix
    sample_rate = args.sample_rate
    classes_num = args.classes_num
    duration = args.duration

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
    rng = np.random.default_rng(135)

    audios_dir = os.path.abspath(os.path.join(dataset_dir))
    packed_hdf5_path = os.path.abspath(os.path.join(workspace_dir, "features"))

    if not suffix.isspace():
        packed_hdf5_path = os.path.join(packed_hdf5_path, f"waveform_{suffix}_seconds.h5")
    else:
        packed_hdf5_path = os.path.join(packed_hdf5_path, "waveform.h5")

    if not os.path.exists(os.path.dirname(packed_hdf5_path)):
        os.makedirs(os.path.dirname(packed_hdf5_path))

    audio_names, audio_paths = [], []

    for root, _, files in os.walk(audios_dir):
        for name in files:
            filepath = os.path.join(root, name)
            audio_names.append(name)
            audio_paths.append(filepath)

    _, lb_to_idx = set_labels(audios_dir)
    # idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}

    meta_dict = {
        "audio_name": np.array(audio_names),
        "audio_path": np.array(audio_paths),
        "target": np.array([lb_to_idx[audio_path.split(os.sep)[-2]] for audio_path in audio_paths]),
        "fold": np.arange(len(audio_names)) % 10 +1
    }
    audios_num = len(meta_dict["audio_name"])

    # force audios to fixed number of samples
    effective_length = floor(sample_rate * duration)

    with h5py.File(packed_hdf5_path, "w") as h5_file:
        h5_file.create_dataset(
            name="audio_name",
            shape=(audios_num,),
            dtype="S80"
        )

        h5_file.create_dataset(
            name="waveform",
            shape=(audios_num, effective_length),
            dtype=np.int16
        )

        h5_file.create_dataset(
            name="target",
            shape=(audios_num, classes_num),
            dtype=np.float32
        )

        h5_file.create_dataset(
            name="fold",
            shape=(audios_num,),
            dtype=np.int32
        )

        for i in range(audios_num):
            audio_name = meta_dict["audio_name"][i]
            fold = meta_dict["fold"][i]
            audio_path = meta_dict["audio_path"][i]

            # random period of audio
            offset = rng.integers(
                    max(librosa.get_duration(filename=audio_path)-duration, 1)
            )

            (audio, _) = librosa.core.load(audio_path, offset=offset, sr=None,
                duration=duration)
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

            h5_file['audio_name'][i] = audio_name.encode()
            h5_file['waveform'][i] = float32_to_int16(audio)
            h5_file['target'][i] = to_one_hot(meta_dict['target'][i], classes_num)
            h5_file['fold'][i] = fold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of dataset."
    )

    parser.add_argument(
        "--workspace_dir",
        type=str,
        required=True,
        help="Directory of your workspace."
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to the append to workspace file. Default ''\
        When provided, it'll change append to the filename _durationprovided_seconds\
        Example: suffix = 1_15, waveform_1_15_seconds."
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Sample rate for load each audio."
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="How much seconds load of each audio."
    )

    parser.add_argument(
        "--classes_num",
        type=int,
        default=8,
        help="Number of classes/directories."
    )

    arguments = parser.parse_args()
    pack_audio_files_to_hdfs(arguments)
