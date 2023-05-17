# pylint: disable=missing-module-docstring
import os
import warnings
import argparse

import numpy as np
import torch
import torch.utils.data as data
import torch.optim as optim

from panns_model import Transfer_Cnn14
from utils.utilities import move_data_to_device, do_mixup, clip_nll
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

def evaluate_model(model, loss_func, validation_loader, device):
    """Evaluate the model and returns a dictionary with the audios
        and classes/classifications for debugging"""

    output_dict = {}
    audio_names = []
    clipwise_outputs = []
    targets = []

    val_loss = 0
    for _, batch_data_dict in enumerate(validation_loader):
        batch_waveform = move_data_to_device(batch_data_dict["waveform"], device)
        batch_audio_name = move_data_to_device(batch_data_dict["audio_name"], device)
        batch_target = move_data_to_device(batch_data_dict["target"], device)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        audio_names.append(batch_audio_name)
        clipwise_outputs.append(batch_output)
        targets.append(batch_target)

        loss = loss_func(batch_output, batch_target)
        val_loss += loss.item() * len(batch_output)

    targets = np.concatenate(targets, axis=0)
    clipwise_outputs = np.concatenate(clipwise_outputs, axis=0)
    audio_names = np.concatenate(audio_names, axis=0)

    val_loss /= len(clipwise_outputs)

    if device == 'cuda':
        y_true = targets.data.cpu().numpy()
        y_pred = clipwise_outputs.data.cpu().numpy()

    else:
        y_true = targets
        y_pred = clipwise_outputs

    class_indices = np.argmax(y_true, axis=-1)
    predict_indexes = np.argmax(y_pred, axis=-1)
    # pylint: disable=invalid-name
    f1 = f1_score(class_indices, predict_indexes, average='macro')
    precision = precision_score(class_indices, predict_indexes, average='macro')
    recall = recall_score(class_indices, predict_indexes, average="macro")
    acc = accuracy_score(class_indices, predict_indexes)

    output_dict = {"audio_name": audio_names, "target": targets, "output": clipwise_outputs}

    return val_loss, f1, precision, recall, acc, output_dict

# pylint: disable=[missing-function-docstring, too-many-arguments, too-many-locals]
def train_model(model, train_loader, validation_loader, tracking_server_uri,
                experiment_name, iterations, audio_augmentation, device, config):

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
                batch_output = model(
                    batch_data_dict["waveform"],
                    batch_data_dict["mixup_lambda"]
                )
                batch_target = do_mixup(batch_data_dict["target"], batch_data_dict["mixup_lambda"])

            else:
                batch_output = model(batch_data_dict["waveform"], None)
                batch_target = batch_data_dict["target"]

            loss = clip_nll(batch_output, batch_target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pylint: disable=[invalid-name, line-too-long]
            val_loss, f1, precision, recall, acc, _ = evaluate_model(model, clip_nll, validation_loader, device)

            print(f"Iteration {iteration+1}- training loss: {loss}. val loss: {val_loss}. f1: {f1}.",
                f"prec val: {precision}. recall val: {recall}. acc val: {acc}")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("train_loss", loss)
            mlflow.log_metric("val_loss", val_loss)

        mlflow.pytorch.log_model(model, artifact_path="models")
        mlflow.set_tags(config)


def run(params):
    iterations = params.model_iterations
    experiment_name = params.experiment_name
    tracking_server_uri = params.tracking_server_uri
    audio_aug = params.audio_augmentation
    batch_size = params.batch_size
    holdout_fold = params.holdout_fold
    num_workers = params.num_workers
    train_path = params.train_path
    pretrained_path = params.pretrained_path
    device = 'cuda' if (params.cuda and torch.cuda.is_available()) else 'cpu'

    # reproducibility
    torch.manual_seed(135)
    torch.cuda.manual_seed(135)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    config = dict(sample_rate=params.audio_sample_rate, window_size=params.audio_window_size,
        hop_size=params.audio_hop_size, mel_bins=params.spectrum_mel_bins, fmin=params.audio_fmin,
        fmax=params.audio_fmax, classes_num=params.classes_num, freeze_base=params.freeze_base)

    train_loader, validation_loader = load_data(train_path, holdout_fold, batch_size, num_workers)
    model = Transfer_Cnn14(**config)
    if len(pretrained_path) == 0:
        print("Training model without Transfer Learning")

    elif not os.path.isfile(pretrained_path):
        raise OSError(f"File {pretrained_path} does not exist")

    else:
        model.load_from_pretrain(pretrained_path)

    train_model(
        model,
        train_loader,
        validation_loader,
        tracking_server_uri,
        experiment_name,
        iterations,
        audio_aug,
        device,
        config
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
        "--pretrained_path",
        type=str,
        default="",
        help="Path of the model trained to do transfer learning"
    )

    parser.add_argument(
        "--freeze_base",
        action="store_false",
        default=True
    )

    parser.add_argument(
        "--classes_num",
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
