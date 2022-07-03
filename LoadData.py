from datasets import load_dataset
import numpy as np
from datasets import Dataset
import argparse
from typing import List
import csv


def get_args():
    # get command arguments
    parser = argparse.ArgumentParser(
        description="Get different information of different datasets"
    )
    parser.add_argument("--ds", nargs=2, type=str, help="name of dataset, split")
    parser.add_argument(
        "--tm", nargs=1, type=str, help="tokenization method: whitespacing, spacy"
    )
    parser.add_argument("--sts", nargs=1, type=str, help="type of stats: simi, len")
    # parser.add_argument(
    #     "--sf", nargs=1, type=str, help="if save the figure: 1, otherwise: 0"
    # )
    args = parser.parse_args()

    return args


def rename_datasets(dataset):
    dataset = dataset.rename_column(dataset.column_names[0], "source")
    dataset = dataset.rename_column(dataset.column_names[1], "target")
    return dataset


def load_data(dataset_name: str, split_: str = "train", p: float = 1):
    version_dic = {
        "cnn_dailymail": "3.0.0",
        "xsum": "1.2.0",
        "wiki_lingua": "english",
        "scitldr_A": "Abstract",
        "scitldr_F": "FullText",
        "billsum": "3.0.0",
    }
    version = version_dic.get(dataset_name)

    if dataset_name == "scitldr_A" or dataset_name == "scitldr_F":
        dataset_name = "scitldr"

    dataset = load_dataset(dataset_name, version, split=split_)

    if dataset_name == "wiki_lingua":
        dataset = dataset.rename_column("article", "source")
    elif dataset_name == "scitldr":
        pass
    else:
        dataset = rename_datasets(dataset)

    # use a certain proportion of samples
    if p != 1:
        n = np.int64(dataset.num_rows * p)
        dataset = dataset.shuffle()
        dataset = Dataset.from_dict(dataset[:n])

    return dataset


def write_csv(filename, data: List):
    with open(filename, "a+", newline="\n", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)


def read_csv(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(len(data)):
        data[i] = list(map(eval, data[i]))

    return data
