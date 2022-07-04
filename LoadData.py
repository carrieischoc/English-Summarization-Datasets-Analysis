import numpy as np
import argparse
import csv
import json
from typing import List
from datasets import load_dataset


def get_args():
    # get command arguments
    parser = argparse.ArgumentParser(
        description="Get different information of different datasets"
    )
    parser.add_argument(
        "-d", "--dataset", nargs=2, type=str, help="name of dataset, split"
    )
    parser.add_argument(
        "-t",
        "--tokenmethod",
        nargs=1,
        type=str,
        help="tokenization method: whitespace, spacy",
    )
    parser.add_argument(
        "-s", "--stats", nargs=1, type=str, help="type of stats: simi, len"
    )
    # parser.add_argument(
    #     "--sf", nargs=1, type=str, help="if save the figure: 1, otherwise: 0"
    # )
    parser.add_argument(
        "-p",
        "--proporofsa",
        nargs="?",
        type=float,
        const=1,
        help="proportion of samples",
    )

    args = parser.parse_args()

    return args


def rename_datasets(dataset):
    dataset = dataset.rename_column(dataset.column_names[0], "source")
    dataset = dataset.rename_column(dataset.column_names[1], "target")
    return dataset


def load_data(dataset_name: str, split: str = "train", data_proportion: float = 1.0):

    with open("datasets_dict.txt") as f:
        dicts = f.read()

    # reconstructing the data as a dictionary
    version_dic = json.loads(dicts)
    version = version_dic.get(dataset_name)

    if dataset_name == "scitldr_A" or dataset_name == "scitldr_F":
        dataset_name = "scitldr"

    dataset = load_dataset(dataset_name, version, split=split)

    if dataset_name == "wiki_lingua":
        dataset = dataset.rename_column("article", "source")
    elif dataset_name == "scitldr":
        pass
    else:
        dataset = rename_datasets(dataset)

    # use a certain proportion of samples
    if data_proportion != 1:
        n = int(dataset.num_rows * data_proportion)
        dataset = dataset.select(np.random.randint(dataset.num_rows, size=n))

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
