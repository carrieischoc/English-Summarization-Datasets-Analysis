from summaries.aligners import RougeNAligner
from inspection import load_data
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from typing import NamedTuple
from datasets import Dataset


def compute_similarity(dataset, n_gram: int = 2) -> NamedTuple:
    """
    Compute mean and maximum similarities of each sentence in summary compared with each article.
    Input: dataset
    Output: mean of mean similarities of summaries, mean of maximum similarities of summaries.
    """

    # use fmeasure to determine similarity
    aligner = RougeNAligner(n=n_gram, optimization_attribute="fmeasure", lang="en")

    similarity = namedtuple("similarity", "mean m_max max")

    mm = []  # mean of mean
    m_max = []  # mean of maximum
    maxi = 0 # maximum similarity of all summary sentences

    for sample in tqdm(dataset):
        # ignore empty source and target
        if sample["target"] == "" or sample["source"] == "": 
            continue
        m = []  # mean
        for aligned_sentence in aligner.extract_source_sentences(
            sample["target"], sample["source"]
        ):
            m.append(aligned_sentence.metric)
        
        mm.append(sum(m) / len(m))
        m_max.append(max(m))
        if maxi < max(m):
            maxi = max(m)

    similarity.mean = sum(mm) / len(mm)
    similarity.m_max = sum(m_max) / len(m_max)
    similarity.max = maxi

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
        f"[{dataset_name}] Mean of maximum similarities of summaries: {similarity.m_max:.2f}."
    )
    print(f"[{dataset_name}] Maximum similarity of all summaries: {similarity.max:.2f}.")


# load data and print stats of cnn_dailymail
load_print("cnn_dailymail", "3.0.0", "train")

# load data and print stats of xsum
# load_print("xsum", "1.2.0", "train")

# load data and print stats of wiki_lingua English
# load_print("wiki_lingua", "english", "train")

# load data and print stats of scitldr
# load_print("scitldr", "Abstract", "train")
# load_print("scitldr", "FullText", "train")

# load data and print stats of billsum
# load_print("billsum", "3.0.0", "train")


