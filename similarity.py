from summaries.aligners import RougeNAligner
from inspection import load_data
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from typing import NamedTuple
from datasets import Dataset
import numpy as np

similarity = namedtuple(
    "similarity", "mean median std max_max mean_max min_min mean_min "
)

def compute_similarity(dataset, n_gram: int = 2) -> NamedTuple:
    """
    Compute mean and maximum similarities of each sentence in summary compared with each article.
    Input: dataset
    Output: mean of mean similarities of summaries, mean of maximum similarities of summaries.
    """

    # use fmeasure to determine similarity
    aligner = RougeNAligner(n=n_gram, optimization_attribute="fmeasure", lang="en")

    mean = []  # mean of mean
    mean_max = []  # mean of maximum
    mean_min = []  # mean of minimum
    maxi = 0  # maximum similarity of all summary sentences
    mini = np.inf  # minimum similarity of all summary sentences

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

        mean.append(np.mean(m))
        mean_max.append(max(m))
        mean_min.append(min(m))
        if maxi < max(m):
            maxi = max(m)
        if mini > min(m):
            mini = min(m)

    similarity.mean = np.mean(mean)
    similarity.median = np.median(mean)
    similarity.std = np.std(mean)
    similarity.mean_max = np.mean(mean_max)
    similarity.mean_min = np.mean(mean_min)
    similarity.max_max = maxi
    similarity.min_min = mini


def load_print(dataset_name: str, version: str, split_: str = "train") -> None:
    dataset = load_data(dataset_name, version, split_)
    if dataset_name == "wiki_lingua":
        dataset = pd.DataFrame(dataset["source"])
        dataset = dataset.rename(columns={"document": "source", "summary": "target"})
        dataset = Dataset.from_pandas(dataset)

    compute_similarity(dataset)

    print(f"[{dataset_name}] [Similarity] Mean: {similarity.mean:.4f}.")
    print(f"[{dataset_name}] [Similarity] Median: {similarity.median:.4f}.")
    print(f"[{dataset_name}] [Similarity] std: {similarity.std:.4f}.")
    print(f"[{dataset_name}] [Similarity] max_max: {similarity.max_max:.4f}.")
    print(f"[{dataset_name}] [Similarity] min_min: {similarity.min_min:.4f}.")
    print(f"[{dataset_name}] [Similarity] mean_max: {similarity.mean_max:.4f}.")
    print(f"[{dataset_name}] [Similarity] mean_min: {similarity.mean_min:.4f}.")


# load data and print stats of cnn_dailymail
# load_print("cnn_dailymail", "3.0.0", "train")

# load data and print stats of xsum
# load_print("xsum", "1.2.0", "train")

# load data and print stats of wiki_lingua English
# load_print("wiki_lingua", "english", "train")

# load data and print stats of scitldr
load_print("scitldr", "Abstract", "train")
# load_print("scitldr", "FullText", "train")

# load data and print stats of billsum
# load_print("billsum", "3.0.0", "train")
