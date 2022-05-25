"""
Segregation step
"""
import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s - %(levelname)s - %(message)s"
)


def main(primary_data, train_data, test_data, test_size, random_state):
    """
    Data segregation. Split primary data to train and test part.

    Args:
        primary_data: primary data path
        train_data: path of train data part
        test_data: path of test data part
        test_size: test size for data spliting
        random_state: random seed for data spliting
    Returns:
        None
    """

    try:
        assert os.path.exists(primary_data)
        logging.info("File was found")

    except AssertionError as error:
        logging.error("File %s not found", primary_data)
        raise error

    primary = pd.read_csv(primary_data)
    logging.info("Primary data shape: %s", primary.shape)
    train, test = train_test_split(
        primary, test_size=test_size, random_state=random_state
    )
    logging.info("Train data shape: %s", train.shape)
    train.to_csv(train_data, index=False)
    logging.info("Test data shape: %s", test.shape)
    test.to_csv(test_data, index=False)


if __name__ == "__main__":

    with open("params.yaml", mode="r", encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    logging.info("Segregation step - Running")
    main(
        config["processing"]["primary_data"],
        config["segregation"]["train_data"],
        config["segregation"]["test_data"],
        config["segregation"]["test_size"],
        config["segregation"]["random_state"],
    )
    logging.info("Segregation step - Completed")
