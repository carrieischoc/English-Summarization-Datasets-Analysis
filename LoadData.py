import argparse
import json
import numpy as np
from typing import List
from datasets import load_dataset


def read_js(filename):
    with open(filename) as f:
        js = json.load(f)

    return js


def get_args():

    # get dataset names
    ds_names = read_js("ds_name_list.json")

    # get command arguments
    parser = argparse.ArgumentParser(
        description="Get different information of different datasets"
    )
    parser.add_argument(
        "-d", "--dataset", nargs=1, type=str, choices=ds_names, help="name of dataset"
    )
    parser.add_argument("--split", nargs=1, type=str, help="split")
    parser.add_argument(
        "-t",
        "--token_method",
        choices=["whitespace", "spacy"],
        nargs=1,
        type=str,
        help="tokenization method: whitespace, spacy",
    )
    parser.add_argument(
        "--stats",
        nargs=1,
        type=str,
        choices=["similarity", "length"],
        help="type of stats: similarity, length",
    )
    # parser.add_argument(
    #     "--sf", nargs=1, type=str, help="if save the figure: 1, otherwise: 0"
    # )
    parser.add_argument(
        "-p",
        "--sample_propor",
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

    version_dic = read_js("ds_version_dict.json")

    # reconstructing the data as a dictionary
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
        dataset = dataset.select(
            np.random.choice(dataset.num_rows, size=n, replace=False)
        )

    return dataset
