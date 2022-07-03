import numpy as np
import pandas as pd
import en_core_web_sm
from tqdm import tqdm
from collections import namedtuple
from LoadData import load_data
from typing import NamedTuple, List


def spacy_token(samples: List[str]) -> NamedTuple:
    """
    Compute number of tokens in each row.
    Input: rows of tokens, number of rows.
    Output: mean, standard deviation and median of tokens.
    """

    stats = namedtuple("stats", "mean median std")

    nlp = en_core_web_sm.load(
        disable=("tok2vec", "tagger", "lemmatizer", "ner")
    )  # Disabling components for only tokenization use.

    # lens = [len(nlp(token)) for token in tqdm(tokens)]
    tokens = list(tqdm(nlp.pipe(samples, n_process=8), total=len(samples)))
    lens = [len(token) for token in (iter(tokens))]

    # write data into .txt file
    # write_csv("length.txt", lens)

    stats.lens = np.array(lens)
    stats.mean = np.mean(lens)
    stats.median = np.median(lens)
    stats.std = np.std(lens)

    return stats


def whitespace_token(samples: List[str]) -> NamedTuple:

    stats = namedtuple("stats", "mean median std lens")

    lens = samples.str.split().str.len()

    # write data into .txt file
    # write_csv("length.txt", lens)

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
    df.replace(to_replace=r"^\s*$", value=np.nan, regex=True, inplace=True)
    df = df.dropna()
    return df


def lens_cal(dataset, tokenization_method: str = "whitespace") -> NamedTuple:

    # Use pandas dataframe to process data and remove samples that contain empty strings.

    if dataset.features["source"]._type == "Value":  # One row of source is one article.
        dataset = dataset.to_pandas()

    elif (
        dataset.features["source"]._type == "Sequence"
    ):  # One row of source is a line in article or combined with article, summary and id.
        dataset = format_tuning(dataset)

    dataset = remove_empty(dataset)

    if tokenization_method == "whitespace":
        stats_src = whitespace_token(dataset["source"])
        stats_tg = whitespace_token(dataset["target"])
    elif tokenization_method == "spacy":
        stats_src = spacy_token(dataset["source"])
        stats_tg = spacy_token(dataset["target"])

    compression_ratio = np.mean(stats_src.lens / stats_tg.lens)

    stats_attr = namedtuple("stats_attr", "mean median std lens")
    stats_all = namedtuple("stats", "src tg compression_ratio SampleNum")
    src = stats_attr(stats_src.mean, stats_src.median, stats_src.std, stats_src.lens)
    tg = stats_attr(stats_tg.mean, stats_tg.median, stats_tg.std, stats_tg.lens)
    stats = stats_all(src, tg, compression_ratio, dataset.shape[0])

    return stats


def print_lens(
    stats, stats_to_compute, dataset_name: str, tokenization_method: str = "whitespace"
) -> None:

    print(f"********{tokenization_method}********")
    if "SampleNum" in stats_to_compute:
        print(
            f"[{dataset_name}] Number of samples of article or summary: {stats.SampleNum}"
        )
    if "mean" in stats_to_compute:
        print(
            f"[{dataset_name}] Mean of article & summary: {stats.src.mean:.2f}, {stats.tg.mean:.2f}"
        )
    if "median" in stats_to_compute:
        print(
            f"[{dataset_name}] Median of article & summary: {stats.src.median:.2f}, {stats.tg.median:.2f}"
        )
    if "std" in stats_to_compute:
        print(
            f"[{dataset_name}] Standard Deviation of article & summary: {stats.src.std:.2f}, {stats.tg.std:.2f}"
        )
    if "compression_ratio" in stats_to_compute:
        print(
            f"[{dataset_name}] ratio of article/summary: {stats.compression_ratio:.2f}"
        )


def get_lens(
    dataset_name: str,
    split: str = "train",
    tokenization_method: str = "whitespace",
    p: float = 1,
) -> NamedTuple:
    dataset = load_data(dataset_name, split, p)
    stats = lens_cal(dataset, tokenization_method)

    return stats


def get_print_lens(
    dataset_name: str,
    split: str = "train",
    tokenization_method: str = "whitespace",
    stats_to_compute: List[str] = [
        "SampleNum",
        "mean",
        "median",
        "std",
        "compression_ratio",
    ],
    p: float = 1,
) -> None:
    stats = get_lens(dataset_name, split, tokenization_method, p)
    print_lens(stats, stats_to_compute, dataset_name, tokenization_method)
