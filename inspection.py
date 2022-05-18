from datasets import load_dataset
import numpy as np
import pandas as pd
import en_core_web_sm
from tqdm import tqdm
from collections import namedtuple
from typing import List, Tuple, NamedTuple


def rename_datasets(dataset):
    dataset = dataset.rename_column(dataset.column_names[0], "source")
    dataset = dataset.rename_column(dataset.column_names[1], "target")
    return dataset


def spacy_token(tokens: List[str]) -> NamedTuple:
    """
    Compute number of tokens in each row.
    Input: rows of tokens, number of rows.
    Output: mean, standard deviation and median of tokens.
    """

    stats = namedtuple("stats", "mean median std")

    lens = [len(nlp(token)) for token in tqdm(tokens)]

    lens = np.array(lens)
    stats.mean = np.mean(lens)
    stats.median = np.median(lens)
    stats.std = np.std(lens)

    return stats


def whitespace_token(tokens: List[str]) -> NamedTuple:

    stats = namedtuple("stats", "mean median std lens")

    lens = tokens.str.split().str.len()

    lens = np.array(lens)
    stats.lens = lens
    stats.mean = np.mean(lens)
    stats.median = np.median(lens)
    stats.std = np.std(lens)

    return stats


def format_tuning(dataset):
    try:
        if (
            dataset.features["source"].feature._type == "Value"
        ):  # One row of source is a line in article.
            dataset = dataset.to_pandas()

    except AttributeError:
        if len(dataset.features["source"].feature) > 1:
            dataset = pd.DataFrame(dataset["source"])
            dataset = dataset.rename(
                columns={"document": "source", "summary": "target"}
            )

    dataset["source"] = dataset["source"].str.join("")
    dataset["target"] = dataset["target"].str.join("")
    return dataset


def stats_cal(
    dataset,
    dataset_name: str,
    tokenization_method: str = "whitespace",
    stats_to_compute: List[str] = [
        "SampleNum",
        "mean",
        "median",
        "std",
        "compression_ratio",
    ],
) -> NamedTuple:

    stats_attr_src = namedtuple("stats_attr", stats_to_compute)
    stats_attr_tg = namedtuple("stats_attr", stats_to_compute)
    stats = namedtuple("stats", "src tg")
    stats.src = stats_attr_src
    stats.tg = stats_attr_tg
    stats.src.SampleNum = dataset.num_rows
    stats.tg.SampleNum = dataset.num_rows

    if dataset.features["source"]._type == "Value":  # One row of source is one article.
        if tokenization_method == "whitespace":
            dataset = dataset.to_pandas()
    elif (
        dataset.features["source"]._type == "Sequence"
    ):  # One row of source is a line in article or combined with article, summary and id.
        dataset = format_tuning(dataset)

    if tokenization_method == "whitespace":
        stats_src = whitespace_token(dataset["source"])
        stats_tg = whitespace_token(dataset["target"])
    elif tokenization_method == "spacy":
        stats_src = spacy_token(dataset["source"])
        stats_tg = spacy_token(dataset["target"])

    if "SampleNum" in stats_to_compute:
        print(
            f"[{dataset_name}] Number of samples of article or summary: {stats.src.SampleNum}"
        )
    if "mean" in stats_to_compute:
        stats.src.mean = stats_src.mean
        stats.tg.mean = stats_tg.mean
        print(
            f"[{dataset_name}] Mean of article & summary: {stats.src.mean:.2f}, {stats.tg.mean:.2f}"
        )
    if "median" in stats_to_compute:
        stats.src.median = stats_src.median
        stats.tg.median = stats_tg.median
        print(
            f"[{dataset_name}] Median of article & summary: {stats.src.median:.2f}, {stats.tg.median:.2f}"
        )
    if "std" in stats_to_compute:
        stats.src.std = stats_src.std
        stats.tg.std = stats_tg.std
        print(
            f"[{dataset_name}] Standard Deviation of article & summary: {stats.src.std:.2f}, {stats.tg.std:.2f}"
        )
    if "compression_ratio" in stats_to_compute:
        stats.src.compression_ratio = np.mean(stats_src.lens / stats_tg.lens)
        stats.tg.compression_ratio = stats.src.compression_ratio
        print(
            f"[{dataset_name}] ratio of article/summary: {stats.src.compression_ratio:.2f}"
        )

    # print(len(stats_tg.lens[np.where(stats_tg.lens == 0)]))
    return stats


def print_stats(
    dataset, dataset_name: str, tokenization_method: str = "whitespace"
) -> None:

    print(f"********{tokenization_method}********")
    stats = stats_cal(dataset, dataset_name)


def load_data(dataset_name: str, version: str, split_: str = "train"):
    dataset = load_dataset(dataset_name, version, split=split_)
    if dataset_name == "wiki_lingua":
        dataset = dataset.rename_column("article", "source")
    elif dataset_name == "scitldr":
        pass
    else:
        dataset = rename_datasets(dataset)

    return dataset


nlp = en_core_web_sm.load(
    disable=("tok2vec", "tagger", "lemmatizer", "ner")
)  # Disabling components for only tokenization use.

# load cnn_dailymail
cnn_train = load_data("cnn_dailymail", "3.0.0", "train")
print_stats(cnn_train, "cnn_dailymail")
# cnn_test = load_dataset("cnn_dailymail", "3.0.0", split="test")
# cnn_valid = load_dataset("cnn_dailymail", "3.0.0", split="validation")


# load xsum
xsum_train = load_data("xsum", "1.2.0", "train")
print_stats(xsum_train, "xsum")
# xsum_test = load_dataset("xsum", "1.2.0", split="test")
# xsum_valid = load_dataset("xsum", "1.2.0", split="validation")

# load wiki_lingua English
# wiki_lingua_train = load_data("wiki_lingua", "english", "train")
# print_stats(wiki_lingua_train,"wiki_lingua")


# load scitldr
scitldr_train = load_data("scitldr", "Abstract", "train")
print_stats(scitldr_train, "scitldr")
# scitldr_test = load_dataset("scitldr", "Abstract", split="test")
# scitldr_valid = load_dataset("scitldr", "Abstract", split="validation")


# load billsum
billsum_train = load_data("billsum", "3.0.0", "train")
print_stats(billsum_train, "billsum")
# billsum_test = load_dataset("billsum", "3.0.0", split="test")
# billsum_valid = load_dataset("billsum", "3.0.0", split="ca_test")
