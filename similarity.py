from summaries.aligners import RougeNAligner
from inspection import load_data
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from typing import NamedTuple
from datasets import Dataset


def compute_similarity(dataset) -> NamedTuple:
    """
    Compute mean and maximum similarities of each sentence in summary compared with each article.
    Input: dataset
    Output: mean of mean similarities of summaries, mean of maximum similarities of summaries.
    """

    # use fmeasure to determine similarity
    aligner = RougeNAligner(n=2, optimization_attribute="fmeasure", lang="en")

    similarity = namedtuple("similarity", "mean max")

    mm = []  # mean of mean
    m_max = []  # mean of maximum

    for sample in tqdm(dataset):
        m = []  # mean
        for aligned_sentence in aligner.extract_source_sentences(
            sample["target"], sample["source"]
        ):
            m.append(aligned_sentence.metric)
        if len(m) != 0:  # ignore results of empty rows
            mm.append(sum(m) / len(m))
            m_max.append(max(m))

    similarity.mean = sum(mm) / len(mm)
    similarity.max = sum(m_max) / len(m_max)

    return similarity


def load_print(dataset_name: str, version: str, split_: str = "train") -> None:
    dataset = load_data(dataset_name, version, split_)
    if dataset_name == "wiki_lingua":
        dataset = pd.DataFrame(dataset["source"])
        dataset = dataset.rename(columns={"document": "source", "summary": "target"})
        dataset = Dataset.from_pandas(dataset)

    similarity = compute_similarity(dataset)

    print(f"[{dataset_name}] Mean similarity of summaries: {similarity.mean:.2f}.")
    print(
        f"[{dataset_name}] Mean of maximum similarities of summaries: {similarity.mean:.2f}."
    )


# load data and print stats of cnn_dailymail
# load_print("cnn_dailymail", "3.0.0", "train")

# load data and print stats of xsum
# load_print("xsum", "1.2.0", "train")

# load data and print stats of wiki_lingua English
load_print("wiki_lingua", "english", "train")

# load data and print stats of scitldr
# load_print("scitldr", "Abstract", "train")
# load_print("scitldr", "FullText", "train")

# load data and print stats of billsum
# load_print("billsum", "3.0.0", "train")
