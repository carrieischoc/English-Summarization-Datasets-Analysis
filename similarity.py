from summaries.aligners import RougeNAligner
from LoadData import load_data
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from datasets import Dataset
import numpy as np
from typing import NamedTuple, List


def compute_similarity(dataset, n_gram: int = 2) -> NamedTuple:
    """
    Compute mean and maximum similarities of each sentence in summary compared with each article.
    Input: dataset
    Output: mean of mean similarities of summaries, mean of maximum similarities of summaries.
    """

    similarity = namedtuple("similarity", "mean max min pos")

    # use fmeasure to determine similarity
    aligner = RougeNAligner(n=n_gram, optimization_attribute="fmeasure", lang="en")

    mean = []  # mean of all summaries per article
    maxi = []  # maximum of all summaries per article
    mini = []  # minimum of all summaries per article
    pos = []  # relative position to the most similar sentences of all summaries

    for sample in tqdm(dataset):
        # ignore empty source and target
        if (
            sample["target"] == ""
            or sample["target"] == []
            or sample["source"] == ""
            or sample["source"] == []
        ):
            continue

        m = []  # mean
        for aligned_sentence in aligner.extract_source_sentences(
            sample["target"], sample["source"]
        ):
            m.append(aligned_sentence.metric)
            pos.append(aligned_sentence.relative_position)

        mean.append(np.mean(m))
        maxi.append(max(m))
        mini.append(min(m))

    similarity.mean = mean
    similarity.max = maxi
    similarity.min = mini
    similarity.pos = pos

    # write data into .txt file
    # write_csv("similarity.txt", mean)
    # write_csv("similarity.txt", maxi)
    # write_csv("similarity.txt", mini)
    # write_csv("position.txt", pos)

    return similarity


def print_simi(stats_to_compute, dataset_name: str, similarity: NamedTuple) -> None:
    if "mean" in stats_to_compute:
        print(
            f"[{dataset_name}] [Similarity] Mean of all means of each article : {np.mean(similarity.mean):.4f}."
        )
        print(
            f"[{dataset_name}] [Similarity] Median of all means of each article: {np.median(similarity.mean):.4f}."
        )
        print(
            f"[{dataset_name}] [Similarity] std of all means of each article: {np.std(similarity.mean):.4f}."
        )

    if "max" in stats_to_compute:
        print(
            f"[{dataset_name}] [Similarity] Max of all maximums of each article: {np.max(similarity.max):.4f}."
        )
        print(
            f"[{dataset_name}] [Similarity] Mean of all maximums of each article: {np.mean(similarity.max):.4f}."
        )

    if "min" in stats_to_compute:
        print(
            f"[{dataset_name}] [Similarity] Min of all minimum of each article: {np.min(similarity.min):.4f}."
        )
        print(
            f"[{dataset_name}] [Similarity] Mean of all minimum of each article: {np.mean(similarity.min):.4f}."
        )


def get_simi(dataset_name: str, split: str = "train", p: float = 1) -> NamedTuple:
    dataset = load_data(dataset_name, split, p)

    if dataset_name == "wiki_lingua":
        dataset = pd.DataFrame(dataset["source"])
        dataset = dataset.rename(columns={"document": "source", "summary": "target"})
        dataset = Dataset.from_pandas(dataset)

    similarity = compute_similarity(dataset)

    return similarity


def get_print_simi(
    dataset_name: str,
    split: str = "train",
    stats_to_compute: List[str] = ["mean", "max", "min"],
    p: float = 1,
) -> None:

    similarity = get_simi(dataset_name, split, p)

    # print similariity stats
    print_simi(stats_to_compute, dataset_name, similarity)
