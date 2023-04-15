"""Provides metric's report of a model for the validation data"""
# pylint:disable=wrong-import-position
import sys
import os
# see below
# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time/14132912#14132912
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
# pylint: disable=import-error
from utils.utilities import set_labels, move_data_to_device, append_to_dict
from utils.dataset import EvaluateSampler, GtzanDataset, collate_fn
import numpy as np
import mlflow
import torch
import torch.utils.data as data
from sklearn.metrics import classification_report

# pylint: disable=missing-function-docstring
def main(params):
    dataset_dir = params.dataset_dir
    workspace = params.workspace
    holdout_fold = params.holdout_fold
    run_id = params.run_id
    batch_size = params.batch_size
    device = 'cuda' if (params.cuda and torch.cuda.is_available()) else 'cpu'

    labels, _ = set_labels(dataset_dir)
    mlflow.set_tracking_uri(params.tracking_server_uri)

    logged_model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")
    sampler = EvaluateSampler(workspace, holdout_fold, batch_size)

    loader = data.DataLoader(
        GtzanDataset(),
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    output_dict = {}

    for batch_data_dict in loader:
        batch_waveform_dict = {"waveform": move_data_to_device(batch_data_dict["waveform"], device)}
        # pylint: disable=line-too-long
        batch_audio_name_dict = {"audio_name": move_data_to_device(batch_data_dict["audio_name"], device)}
        batch_target_dict = {"target": move_data_to_device(batch_data_dict["target"], device)}

        with torch.no_grad():
            logged_model.eval()
            batch_output = logged_model(batch_waveform_dict["waveform"])

        append_to_dict(output_dict, 'audio_name', batch_audio_name_dict['audio_name'])
        append_to_dict(output_dict, 'clipwise_output', batch_output['clipwise_output'])
        append_to_dict(output_dict, 'target', batch_target_dict['target'])

    # pylint: disable=consider-iterating-dictionary
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    if device == 'cuda':
        y_true = output_dict['target'].data.cpu().numpy()
        y_pred = output_dict['clipwise_output'].data.cpu().numpy()

    else:
        y_true = output_dict['target']
        y_pred = output_dict['clipwise_output']

    predict_indexes = np.argmax(y_true, axis=-1)
    class_indices = np.argmax(y_pred, axis=-1)
    print(classification_report(
        class_indices,
        predict_indexes,
        target_names=labels,
        zero_division=0
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Original Path used to prepare the audios for PANNs"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="hdf5 path used to train PANNs models"
    )

    parser.add_argument(
        "--tracking_server_uri",
        type=str,
        default="http://127.0.0.1:5000"
    )

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="ID run on tracking server"
    )

    parser.add_argument(
        "--holdout_fold",
        type=str,
        default="8",
        help="Path to compute the metrics. Default value is the validation fold"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Path to compute the metrics. Default value is the validation fold"
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    main(args)
