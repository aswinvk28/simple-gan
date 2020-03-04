import os, sys

from IE.MNISTGenerator import MNISTGenerator
import argparse
from network.utils import Logger
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='', allow_abbrev=False)
    parser.add_argument(
        '-m', '--model',
        help='IR model',
        type=str,
        default="",
        required=True
    )

    return parser.parse_args()

def vectors_to_images(vectors):
    return vectors.reshape(vectors.shape[0], 1, 28, 28)

if __name__ == "__main__":

    args = parse_args()
    logger = Logger(model_name="Generator", data_name="MNIST")

    gen = MNISTGenerator()
    gen.load_model(args.model)
    num_test_samples = 16
    gen.async_inference(np.random.normal(0,1,(num_test_samples,100)))
    gen.wait()
    output = gen.extract_output()
    output = vectors_to_images(output)
    logger.log_images(
        output, num_test_samples, 1, 1, 16
    )