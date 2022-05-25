"""
Processing step
"""
import logging
import os

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s - %(levelname)s - %(message)s"
)


def remove_spaces(raw_data: str, primary_data: str):
    """
    Remove all whitespaces from csv file.

    Args:
        raw_data: Raw data path
        primary_data: Primary data path
    Returns:
        None
    """

    try:
        assert os.path.exists(raw_data)
        logging.info("File was found")

    except AssertionError as error:
        logging.error("File %s not found", raw_data)
        raise error

    with open(raw_data, mode="r", encoding="utf8") as raw_file:
        lines = raw_file.readlines()

    lines = [line.replace(" ", "") for line in lines]

    logging.info("Number of records: %s", len(lines))

    with open(primary_data, mode="w", encoding="utf8") as primary_file:
        primary_file.writelines(lines)


def main(params: dict):
    """
    Main function for running step.

    Args:
        params: dictionary with config data.
    Returs:
        None
    """
    logging.info("Processing step - Running")
    remove_spaces(params["raw_data"], params["primary_data"])
    logging.info("Processing step - Completed")


if __name__ == "__main__":

    with open("params.yaml", mode="r", encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config["processing"])
