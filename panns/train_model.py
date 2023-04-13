# pylint: disable=missing-module-docstring
import os

import warnings
import argparse

import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim

from panns_model import Transfer_Cnn14
from utils.utilities import move_data_to_device, do_mixup, append_to_dict, clip_nll
from utils.dataset import GtzanDataset, TrainSampler, EvaluateSampler, collate_fn

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import mlflow

def load_data(hdf5_path, holdout_fold, batch_size, num_workers=0):
    """Load an hdf5 file and prepare train data and validation data to be iterated"""
    dataset = GtzanDataset()

    train_sampler = TrainSampler(hdf5_path, holdout_fold, batch_size)
    validation_sampler = EvaluateSampler(hdf5_path, holdout_fold, batch_size)

    train_loader = data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    validation_loader = data.DataLoader(
        dataset=dataset,
        batch_sampler=validation_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, validation_loader

# pylint: disable=missing-function-docstring
def evaluate_model(model, loss_func, validation_loader, device):
    output_dict = {}

    val_loss = 0
    for _, batch_data_dict in enumerate(validation_loader):
        batch_waveform_dict = {"waveform": move_data_to_device(batch_data_dict["waveform"], device)}
        # pylint: disable=line-too-long
        batch_audio_name_dict = {"audio_name": move_data_to_device(batch_data_dict["audio_name"], device)}
        batch_target_dict = {"target": move_data_to_device(batch_data_dict["target"], device)}

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform_dict["waveform"])

        append_to_dict(output_dict, 'audio_name', batch_audio_name_dict['audio_name'])
        append_to_dict(output_dict, 'clipwise_output', batch_output['clipwise_output'])
        append_to_dict(output_dict, 'target', batch_target_dict['target'])

        loss = loss_func(batch_output, batch_target_dict)
        val_loss += loss.item() * len(batch_output['clipwise_output'])

    # pylint: disable=consider-iterating-dictionary
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    val_loss /= len(output_dict['clipwise_output'])

    if device == 'cuda':
        y_true = output_dict['target'].data.cpu().numpy()
        y_pred = output_dict['clipwise_output'].data.cpu().numpy()

    else:
        y_true = output_dict['target']
        y_pred = output_dict['clipwise_output']

    predict_indexes = np.argmax(y_true, axis=-1)
    class_indices = np.argmax(y_pred, axis=-1)
    # pylint: disable=[invalid-name, line-too-long]
    f1 = f1_score(predict_indexes, class_indices, average='weighted')
    precision = precision_score(predict_indexes, class_indices, average='weighted')
    recall = recall_score(predict_indexes, class_indices, average='weighted')
    acc = accuracy_score(predict_indexes, class_indices)

    return val_loss, f1, precision, recall, acc

# pylint: disable=[missing-function-docstring, too-many-arguments, too-many-locals]
def train_model(model, train_loader, validation_loader, tracking_server_uri,
                experiment_name, iterations, audio_augmentation, device):

    mlflow.set_tracking_uri(tracking_server_uri)
    mlflow.set_experiment(experiment_name)

    warnings.filterwarnings("ignore")

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    with mlflow.start_run():
        mlflow.set_tag("model", "PANNs")

        for iteration, batch_data_dict in enumerate(train_loader):
            if iteration == iterations:
                break

            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            model.train()

            if audio_augmentation:
                batch_output_dict = model(
                    batch_data_dict["waveform"],
                    batch_data_dict["mixup_lambda"]
                )

                batch_target_dict = {
                    "target": do_mixup(batch_data_dict["target"], batch_data_dict["mixup_lambda"])
                }

            else:
                batch_output_dict = model(batch_data_dict["waveform"], None)

                batch_target_dict = {"target": batch_data_dict["target"]}

            loss = clip_nll(batch_output_dict, batch_target_dict)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pylint: disable=[invalid-name, line-too-long]
            val_loss, f1, precision, recall, acc = evaluate_model(model, clip_nll, validation_loader, device)

            print(f"Iteration {iteration+1}- training loss: {loss}. val loss: {val_loss}. f1: {f1}.\
                prec val: {precision}. recall val: {recall}. acc val: {acc}")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

        mlflow.pytorch.log_model(model, artifact_path="models")


def run(params):
    iterations = params.model_iterations
    experiment_name = params.experiment_name
    tracking_server_uri = params.tracking_server_uri
    audio_aug = params.audio_augmentation
    batch_size = params.batch_size
    holdout_fold = params.holdout_fold
    num_workers = params.num_workers
    train_path = params.train_path
    device = 'cuda' if (params.cuda and torch.cuda.is_available()) else 'cpu'

    # reproducibility
    torch.manual_seed(135)
    torch.cuda.manual_seed(135)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    config = dict(sample_rate=params.audio_sample_rate, window_size=params.audio_window_size,
        hop_size=params.audio_hop_size, mel_bins=params.spectrum_mel_bins, fmin=params.audio_fmin,
        fmax=params.audio_fmax, classes_num=params.num_classes)

    train_loader, validation_loader = load_data(train_path, holdout_fold, batch_size, num_workers)
    model = Transfer_Cnn14(**config)
    train_model(
        model,
        train_loader,
        validation_loader,
        tracking_server_uri,
        experiment_name,
        iterations,
        audio_aug,
        device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_iterations",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--tracking_server_uri",
        type=str,
        default="http://127.0.0.1:5000"
    )

    parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=32000
    )

    parser.add_argument(
        "--audio_window_size",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--audio_hop_size",
        type=int,
        default=320
    )

    parser.add_argument(
        "--audio_fmin",
        type=int,
        default=50
    )

    parser.add_argument(
        "--audio_fmax",
        type=int,
        default=14000
    )

    parser.add_argument(
        "--audio_augmentation",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--spectrum_mel_bins",
        type=int,
        default=64
    )

    parser.add_argument(
        "--train_path",
        type=str
    )

    parser.add_argument(
        "--holdout_fold",
        type=str,
        default="8",
        help="Fold to be used for validation"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    run(args)
