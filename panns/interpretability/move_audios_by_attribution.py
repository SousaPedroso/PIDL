"""The initial idea for this module is move audios with negative or low contribution
to the class they belong to a path not used for classification. The DeepLift is being used for this,
but there's no special reason. Feel free to test other algorithms.
"""
import os
import sys
from argparse import ArgumentParser
import numpy as np
import mlflow
import torch
from captum.attr import DeepLift
from interpretability_utilities import load_workspace_file

# pylint: disable=[missing-function-docstring, too-many-locals]
def run(params):
    batch_size = params.batch_size
    workspace = params.workspace
    dataset_dir = params.dataset_dir
    audio_class = params.audio_class
    discard_path = params.discard_path
    threshold = params.threshold
    run_id = params.run_id
    device = 'cuda' if (params.cuda and torch.cuda.is_available()) else 'cpu'

    mlflow.set_tracking_uri(f"{params.tracking_server_uri}:5000")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")

    deep_lift = DeepLift(model)

    i = 1
    while True:
        try:
            inp_data, _, labels, _, _, names = load_workspace_file(workspace, i,
                dataset_dir, device)

            # pylint: disable=used-before-assignment
            indexes = torch.where(torch.argmax(labels, dim=1) == lb_to_idx[audio_class])[0]
            if len(indexes) != 0:
                audios_list = torch.cat((audios_list, inp_data[indexes]), dim=0)
                names_list = np.concatenate((names_list, names[indexes]), axis=0)
            else:
                break

        except NameError:
            audios_list, _, labels, _, lb_to_idx, names_list = load_workspace_file(workspace,
                i, dataset_dir, device)

            indexes = torch.where(torch.argmax(labels, dim=1) == lb_to_idx[audio_class])[0]
            audios_list = audios_list[indexes]
            names_list = names_list[indexes]

        finally:
            i += 1

    save_path = os.path.join(discard_path, audio_class)
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        # pylint: disable=broad-exception-caught
        except Exception as exc:
            print(exc)
            sys.exit()

    for i in range(0, len(audios_list)-batch_size, batch_size):
        attr = deep_lift.attribute(audios_list[i:i+batch_size], target=lb_to_idx[audio_class])
        mean_attr = torch.mean(attr, dim=1)

        for j, attr in enumerate(mean_attr):
            if attr <= threshold:
                os.rename(os.path.join(dataset_dir, audio_class, names_list[i+j].decode()),
                    os.path.join(save_path, names_list[i+j].decode()))

        del mean_attr
        if device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of audios to compute each time attributions. Be aware to high memory consume"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        help="H5 file full path containing the data for training and validation the model.",
        required=True
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory of dataset."
    )

    parser.add_argument(
        "--audio_class",
        type=str,
        help="Class to compute attributions",
        default="others"
    )

    parser.add_argument(
        "--discard_path",
        type=str,
        help="Path to move audios with attribution less than the defined threshold",
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="Minimum value for attribution for an audio do not be moved to discard path"
    )

    parser.add_argument(
        "--tracking_server_uri",
        type=str,
        default="http://127.0.0.1"
    )

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="ID run on tracking server"
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False
    )

    flags = parser.parse_args()
    run(flags)
