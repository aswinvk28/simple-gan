import os, sys
sys.path.append("/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/gan-network/")
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
    parser.add_argument(
        '-b', '--batch',
        help='Batch Size',
        type=int,
        default=1,
        required=True
    )
    parser.add_argument(
        '-mn', '--model_name',
        help='Model name',
        type=str,
        default="",
        required=True
    )
    parser.add_argument(
        '-dn', '--data_name',
        help='Data name',
        type=str,
        default="",
        required=True
    )
    parser.add_argument(
        '-n', '--num',
        help='Number',
        type=int,
        default="",
        required=True
    )

    return parser.parse_args()

def vectors_to_images(vectors):
    return vectors.reshape(vectors.shape[0], 1, 28, 28)

if __name__ == "__main__":

    args = parse_args()
    logger = Logger(model_name=args.model_name, data_name=args.data_name)

    gen = MNISTGenerator()
    gen.load_model(args.model)
    num_test_samples = args.batch
    dist = np.load('dist'+args.num.__str__()+'.npy')
    gen.async_inference(dist)
    gen.wait()
    output = gen.extract_output()
    output = vectors_to_images(output)
    np.save('output'+args.num.__str__()+'.npy', output)
    logger.log_images(
        output, num_test_samples, 1, 1, num_test_samples
    )