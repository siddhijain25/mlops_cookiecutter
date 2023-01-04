# -*- coding: utf-8 -*-
import glob
import io
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

# set filepaths
input_filepath = "C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/raw/"
output_filepath = "C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/processed/"


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # set file paths
    train_files = glob.glob(
        r"C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/raw/train*.npz"
    )
    test_files = glob.glob(
        r"C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/raw/test*.npz"
    )

    # initialize X and y arrays for training
    train_images = []
    train_labels = []
    for file in train_files:
        data = np.load(file)
        train_images.append(data["images"])
        train_labels.append(data["labels"])
    train_images = np.concatenate((train_images), axis=0)
    train_labels = np.concatenate((train_labels), axis=0)

    # initialize X and y arrays for testing
    test_images = []
    test_labels = []
    for file in test_files:
        data = np.load(file)
        test_images.append(data["images"])
        test_labels.append(data["labels"])
    test_images = np.concatenate((test_images), axis=0)
    test_labels = np.concatenate((test_labels), axis=0)

    # convert training data to appropriate combined tensor format
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()

    # convert test data to appropriate combined tensor format
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).long()

    # save the tensors for train and test
    torch.save(
        test_images,
        "C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/processed/test_images.pt",
    )
    torch.save(
        test_labels,
        "C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/processed/test_labels.pt",
    )
    buffer = io.BytesIO()
    torch.save(test_images, buffer)
    torch.save(test_labels, buffer)
    torch.save(
        train_images,
        "C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/processed/train_images.pt",
    )
    torch.save(
        train_labels,
        "C:/Users/Siddhi/Desktop/mlops_cookiecutter/data/processed/train_labels.pt",
    )
    buffer = io.BytesIO()
    torch.save(test_images, buffer)
    torch.save(test_labels, buffer)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
