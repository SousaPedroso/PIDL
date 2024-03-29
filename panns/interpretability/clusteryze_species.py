"""The initital idea for this module is to determine the optimal number of clusters
for OtherBirds class through reduced spectrograms features given what a model learned,
but it can be applied for all the classes
"""
import os
import sys
import shutil
from argparse import ArgumentParser
from interpretability_utilities import evaluate_inputs
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# pylint: disable=[missing-function-docstring, too-many-locals]
def run(params):
    audios_path = params.audios_path
    audio_class = params.audio_class
    output_path = params.output_path
    sample_rate = params.audio_sample_rate
    duration = params.audio_duration
    run_id = params.run_id

    eps = params.dbscan_eps
    min_samples = params.dbscan_min_samples
    metric = params.dbscan_metric
    algorithm = params.dbscan_algorithm
    leaf_size = params.dbscan_leaf_size
    dbscan_p = params.dbscan_p
    n_jobs = params.dbscan_n_jobs

    show_clusters = params.show

    mlflow.set_tracking_uri(f"{params.tracking_server_uri}:5000")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/models")

    audios_evaluation, audios_name  = evaluate_inputs(model, audios_path, audio_class,
            sample_rate, duration)
    audios_evaluation = audios_evaluation.detach().numpy()

    pca = PCA(n_components=2, random_state=135)
    spectrogram_pca = pca.fit_transform(audios_evaluation)

    # pylint: disable=too-many-function-args
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=None,
            algorithm=algorithm, leaf_size=leaf_size, p=dbscan_p,
            n_jobs=n_jobs).fit(spectrogram_pca)

    for i, label  in enumerate(dbscan.labels_):
        save_path = os.path.join(output_path, f"{audio_class}_{label}")
        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            # pylint: disable=broad-exception-caught
            except Exception as exc:
                print(f"DirectoryCreationError: {exc}")
                sys.exit()

        shutil.copy(audios_name[i][0], os.path.join(save_path, audios_name[i][1]))

    if show_clusters:
        sns.scatterplot(x=spectrogram_pca[:, 0], y=spectrogram_pca[:, 1],
            hue=dbscan.labels_, palette="deep")
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--audios_path",
        type=str,
        help="Audios path",
        required=True
    )

    parser.add_argument(
        "--audio_class",
        type=str,
        help="Class to analyze and make clusters",
        default="others"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path to copy the audios splited by the clusters",
        required=True
    )

    parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=24000,
        help="Sample rate for load each audio."
    )

    parser.add_argument(
        "--audio_duration",
        type=float,
        default=1.0,
        help="How much seconds load of each audio."
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

    # dbscan parameters
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=5
    )

    parser.add_argument(
        "--dbscan_metric",
        type=str,
        default="euclidean",
    )

    parser.add_argument(
        "--dbscan_algorithm",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--dbscan_leaf_size",
        type=int,
        default=30
    )

    parser.add_argument(
        "--dbscan_p",
        type=int
    )

    parser.add_argument(
        "--dbscan_n_jobs",
        type=int
    )

    parser.add_argument(
        "--show",
        action="store_false",
        default=True,
        help="Show clusters obtained"
    )

    flags = parser.parse_args()
    run(flags)
