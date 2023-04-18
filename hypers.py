import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Neural Image Captioning model training.", description="A training routine that trains a NIC model."
    )
    parser.add_argument(
        "-w",
        "--word-frequency-thresh",
        type=int,
        default=5,
        help="Least amount of times a word must appear across all captions in order to"
        " be included in the vocabulary bag.",
        metavar="",
    )
    parser.add_argument(
        "-u",
        "--unknown-thresh",
        type=int,
        default=3,
        help='Amount of unknown ("<UNK>") are allowed in a sentence.',
        metavar="",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=int,
        default=512,
        help="Amount of features the encoder will output and the decoder will accept.",
        metavar="",
    )
    parser.add_argument(
        "-H",
        "--hidden-state-size",
        type=int,
        default=512,
        help="Hidden state sizes to run random grid search on.",
        metavar="",
    )
    parser.add_argument("-l", "--lstm-layers", type=int, default=1, help="Amount of LSTM cells to use.", metavar="")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Amount of observations in each batch when training.",
        metavar="",
    )
    parser.add_argument(
        "-L", "--learning-rate", type=float, default=1e-4, help="Initial learning rate for the model.", metavar=""
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Amount of times the dataset will be processed for training.",
        metavar="",
    )
    args = parser.parse_known_args()[0]
    return args
