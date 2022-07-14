from collections import namedtuple
from typing import NamedTuple, List
from tqdm import tqdm
import numpy as np
import pandas as pd
import en_core_web_sm


from LoadData import load_data


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

    tokens = list(tqdm(nlp.pipe(samples, n_process=8), total=len(samples)))
    lens = [len(token) for token in (iter(tokens))]

    stats.lens = np.array(lens)
    stats.mean = np.mean(lens)
    stats.median = np.median(lens)
    stats.std = np.std(lens)

    return stats


def whitespace_token(samples: List[str]) -> NamedTuple:

    stats = namedtuple("stats", "mean median std lens")

    # split the sentences using pandas str whitespace split
    # get the tokens length of each row of sample
    lens = samples.str.split().str.len()

    stats.lens = lens
    stats.mean = np.mean(lens)
    stats.median = np.median(lens)
    stats.std = np.std(lens)

    return stats


def format_tuning(dataset):
    """
    Input: a huggingface dataset with feature type of Sequence
    Output: split features as source and target and combine sentences
    """
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

    mean_ratio = np.mean(stats_src.lens / stats_tg.lens)

    stats_attr = namedtuple("stats_attr", "mean median std lens")
    stats_all = namedtuple("stats", "src tg mean_ratio SampleNum")
    src = stats_attr(stats_src.mean, stats_src.median, stats_src.std, stats_src.lens)
    tg = stats_attr(stats_tg.mean, stats_tg.median, stats_tg.std, stats_tg.lens)
    stats = stats_all(src, tg, mean_ratio, dataset.shape[0])

    return stats


def print_lens(
    stats: NamedTuple,
    stats_to_compute: List[str],
    dataset_name: str,
    tokenization_method: str = "whitespace",
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
    if "mean_ratio" in stats_to_compute:
        print(
            f"[{dataset_name}] mean of compression ratios of article/summary: {stats.mean_ratio:.2f}"
        )


def get_lens(
    dataset_name: str,
    split: str = "train",
    tokenization_method: str = "whitespace",
    data_proportion: float = 1.0,
) -> NamedTuple:
    dataset = load_data(dataset_name, split, data_proportion)
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
        "mean_ratio",
    ],
    data_proportion: float = 1.0,
) -> None:
    stats = get_lens(dataset_name, split, tokenization_method, data_proportion)
    print_lens(stats, stats_to_compute, dataset_name, tokenization_method)
