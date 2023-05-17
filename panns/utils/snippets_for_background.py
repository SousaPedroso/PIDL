""" Script for generating snippets from long-duration records of background"""
import os
import sys
import argparse
import threading
import librosa
import soundfile as sf

# pylint: disable=missing-function-docstring
def main(flags):
    audio_duration = flags.duration
    audios_sample_rate = flags.sample_rate
    audios_input_path = os.path.normpath(flags.input)
    audios_output_path = os.path.normpath(flags.output)
    audios_background_list = flags.backgrounds if len(flags.backgrounds) != 0 \
        else os.listdir(audios_input_path)
    audios_snippets = flags.snippets

    window = round(audio_duration / 2, 3)
    def process_audio(audios: list, background: str, snippets: int):
        """Receives a list of audios, process and save"""

        for audio in audios:

            for snippet in range(snippets):
                signal, _ = librosa.load(os.path.join(audios_input_path, background, audio),
                    sr=audios_sample_rate, duration=audio_duration, offset=round(snippet*window, 3))

                if not os.path.exists(audios_output_path):
                    try:
                        os.makedirs(audios_output_path)
                    # pylint: disable=broad-exception-caught
                    except Exception as exc:
                        print(f"DirectoryCreationError: {exc}")
                        sys.exit()

                new_background_path = os.path.join(audios_output_path, background)

                if not os.path.exists(new_background_path):
                    try:
                        os.makedirs(new_background_path)
                    # pylint: disable=broad-exception-caught
                    except Exception as exc:
                        print(f"DirectoryCreationError: {exc}")
                        sys.exit()

                audio_name = audio[:len(audio)-4]

                sf.write(
                    os.path.join(new_background_path,
                        f"{audio_name}_snippet_{snippet}_window_{window}.wav"),
                    signal,
                    samplerate=audios_sample_rate
                )

    threads = [threading.Thread(
                            target=process_audio,
                            args=(os.listdir(os.path.join(audios_input_path, background)),
                                background, audios_snippets)
                            )
                for background in audios_background_list] # 1 thread for each class

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Duration to load each snippet",
        default=0.15
    )

    parser.add_argument(
        "--sample_rate",
        "-sr",
        type=int,
        help="Sample rate to load each snippet",
        default=24000
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input path to load the audios",
        required=True
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path to save the audios",
        required=True
    )

    parser.add_argument(
        "--backgrounds",
        "-b",
        nargs="*",
        help="Backgrounds to create the snippets",
        default=[]
    )

    parser.add_argument(
        "--snippets",
        "-sn",
        type=int,
        help="Number of snippets to be created for each audio",
        default=125,
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
