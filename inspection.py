import numpy as np
import pandas as pd
import en_core_web_sm
from tqdm import tqdm
from collections import namedtuple
from typing import List, NamedTuple
from LoadData import *


def spacy_token(samples: List[str]) -> NamedTuple:
    """
    Compute number of tokens in each row.
    Input: rows of tokens, number of rows.
    Output: mean, standard deviation and median of tokens.
    """

    stats = namedtuple("stats", "mean median std")

    # lens = [len(nlp(token)) for token in tqdm(tokens)]
    tokens = list(tqdm(nlp.pipe(samples, n_process=8), total=len(samples)))
    lens = [len(token) for token in (iter(tokens))]

    stats.lens = np.array(lens)
    stats.mean = np.mean(lens)
    stats.median = np.median(lens)
    stats.std = np.std(lens)

    return stats


def whitespace_token(samples: List[str]) -> NamedTuple:

    stats = namedtuple("stats", "mean median std lens")

    lens = samples.str.split().str.len()

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

def remove_empty(df):
    df.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True)
    df = df.dropna()
    return df


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

    # Use pandas dataframe to process data and remove samples that contain empty strings. 

    if dataset.features["source"]._type == "Value":  # One row of source is one article.
            dataset = dataset.to_pandas()
            
    elif (
        dataset.features["source"]._type == "Sequence"
    ):  # One row of source is a line in article or combined with article, summary and id.
        dataset = format_tuning(dataset)
    
    dataset = remove_empty(dataset)
    stats.src.SampleNum = dataset.shape[0]
    stats.tg.SampleNum = stats.src.SampleNum
    
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
    stats = stats_cal(dataset, dataset_name, tokenization_method)


def load_print(dataset_name: str, split_: str = "train") -> None:
    dataset = load_data(dataset_name, split_)
    print_stats(dataset, dataset_name) # whitespace tokenization
    print_stats(dataset, dataset_name, "spacy") # spacy tokenization


if __name__ == '__main__':
    nlp = en_core_web_sm.load(
        disable=("tok2vec", "tagger", "lemmatizer", "ner")
    )  # Disabling components for only tokenization use.

    # load data and print stats 
    load_print(args.ds[0], args.ds[1])
