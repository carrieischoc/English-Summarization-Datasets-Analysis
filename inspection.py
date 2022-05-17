from datasets import load_dataset
import numpy as np
import pandas as pd
import en_core_web_sm
from tqdm import tqdm

def rename_datasets(dataset):
    dataset = dataset.rename_column(dataset.column_names[0],"source")
    dataset = dataset.rename_column(dataset.column_names[1],"target")
    return dataset

def spacy_token(tokens):
    """
    Compute number of tokens in each row.
    Input: rows of tokens, number of rows.
    Output: mean, standard deviation and median of tokens.
    """

    lens = [len(nlp(token)) for token in tqdm(tokens)]

    lens = np.array(lens)
    mean = np.mean(lens)
    median = np.median(lens)
    std = np.std(lens)

    return mean, median, std

def whitespace_token(tokens,n):
    lens = tokens.str.split().str.len()
    lens = np.array(lens)
    mean = np.mean(lens)
    median = np.median(lens)
    std = np.std(lens)

    return mean, median, std

def format_tuning(dataset):
    try:
        if dataset.features["source"].feature._type == "Value":
            dataset = dataset.to_pandas()
            
    except:
        if len(dataset.features["source"].feature) > 1:
            dataset = pd.DataFrame(dataset["source"])
            dataset = dataset.rename(columns={"document":"source","summary":"target"})

    dataset["source"] = dataset["source"].str.join("")
    dataset["target"] = dataset["target"].str.join("")
    return dataset

def stats_cal_1(dataset):
    num_samples = dataset.num_rows
    src_stats_list = []
    tg_stats_list = []

    if dataset.features["source"]._type == "Value":
        dataset = dataset.to_pandas()
    elif dataset.features["source"]._type == "Sequence":
        dataset = format_tuning(dataset)

    src_mean, src_median, src_std = whitespace_token(dataset["source"],num_samples)
    tg_mean, tg_median, tg_std = whitespace_token(dataset["target"],num_samples)

    print("Number of samples: ",num_samples)
    print("mean, median and standard deviation of number of tokens of Source: ",src_mean, src_median, src_std)
    print("mean, median and standard deviation of number of tokens of Source: ",tg_mean, tg_median, tg_std)

    src_stats_list = src_stats_list + [src_mean, src_median, src_std]
    tg_stats_list = tg_stats_list + [tg_mean, tg_median, tg_std]
    return src_stats_list, tg_stats_list

def stats_cal_2(dataset):
    num_samples = dataset.num_rows
    src_stats_list = []
    tg_stats_list = []

    if dataset.features["source"]._type == "Value":
        pass
    elif dataset.features["source"]._type == "Sequence":
        dataset = format_tuning(dataset)

    src_mean, src_median, src_std = spacy_token(dataset["source"])
    tg_mean, tg_median, tg_std = spacy_token(dataset["target"])

    print("Number of samples: ",num_samples)
    print("mean, median and standard deviation of number of tokens of Source: ",src_mean, src_median, src_std)
    print("mean, median and standard deviation of number of tokens of Source: ",tg_mean, tg_median, tg_std)

    src_stats_list = src_stats_list + [src_mean, src_median, src_std]
    tg_stats_list = tg_stats_list + [tg_mean, tg_median, tg_std]
    return src_stats_list, tg_stats_list

# load cnn_dailymail
cnn_train = load_dataset("cnn_dailymail", "3.0.0", split="train")
cnn_test = load_dataset("cnn_dailymail", "3.0.0", split="test")
cnn_valid = load_dataset("cnn_dailymail", "3.0.0", split="validation")
cnn_train = rename_datasets(cnn_train)
cnn_test = rename_datasets(cnn_test)
cnn_valid = rename_datasets(cnn_valid)

# load xsum
xsum_train = load_dataset("xsum", "1.2.0", split="train")
xsum_test = load_dataset("xsum", "1.2.0", split="test")
xsum_valid = load_dataset("xsum", "1.2.0", split="validation")
xsum_train = rename_datasets(xsum_train)
xsum_test = rename_datasets(xsum_test)
xsum_valid = rename_datasets(xsum_valid)

# load wiki_lingua English
wiki_lingua_train = load_dataset("wiki_lingua", "english", split="train")
wiki_lingua_train = wiki_lingua_train.rename_column("article","source")

# load scitldr
scitldr_train = load_dataset("scitldr", "Abstract", split="train")
scitldr_test = load_dataset("scitldr", "Abstract", split="test")
scitldr_valid = load_dataset("scitldr", "Abstract", split="validation")


# load billsum
billsum_train = load_dataset("billsum", "3.0.0", split="train")
billsum_test = load_dataset("billsum", "3.0.0", split="test")
billsum_valid = load_dataset("billsum", "3.0.0", split="ca_test")
billsum_train = rename_datasets(billsum_train)
billsum_test = rename_datasets(billsum_test)
billsum_valid = rename_datasets(billsum_valid)

nlp = en_core_web_sm.load(disable=("tok2vec", "tagger", "lemmatizer", "ner"))

print("******Whitespace tokenization******")
cnn_stats_src1, cnn_stats_tg1 = stats_cal_1(cnn_train)
print("[CNN] ratio of article/summary: ", np.array(cnn_stats_src1)/np.array(cnn_stats_tg1))
xsum_stats_src1, xsum_stats_tg1 = stats_cal_1(xsum_train)
print("[XSUM] ratio of article/summary: ", np.array(xsum_stats_src1)/np.array(xsum_stats_tg1))
scitldr_stats_src1, scitldr_stats_tg1 = stats_cal_1(scitldr_train)
print("[scitldr] ratio of article/summary: ", np.array(scitldr_stats_src1)/np.array(scitldr_stats_tg1))
wiki_stats_src1, wiki_stats_tg1 = stats_cal_1(wiki_lingua_train)
print("[wiki] ratio of article/summary: ", np.array(wiki_stats_src1)/np.array(wiki_stats_tg1))
bill_stats_src1, bill_stats_tg1 = stats_cal_1(billsum_train)
print("[Billsum] ratio of article/summary: ", np.array(bill_stats_src1)/np.array(bill_stats_tg1))

print("******Spacy tokenization******")
cnn_stats_src2, cnn_stats_tg2 = stats_cal_2(cnn_train)
print("[CNN] ratio of article/summary: ", np.array(cnn_stats_src2)/np.array(cnn_stats_tg2))
xsum_stats_src2, xsum_stats_tg2 = stats_cal_2(xsum_train)
print("[XSUM] ratio of article/summary: ", np.array(xsum_stats_src2)/np.array(xsum_stats_tg2))
scitldr_stats_src2, scitldr_stats_tg2 = stats_cal_2(scitldr_train)
print("[scitldr] ratio of article/summary: ", np.array(scitldr_stats_src2)/np.array(scitldr_stats_tg2))
wiki_stats_src2, wiki_stats_tg2 = stats_cal_2(wiki_lingua_train)
print("[wiki] ratio of article/summary: ", np.array(wiki_stats_src2)/np.array(wiki_stats_tg2))
bill_stats_src2, bill_stats_tg2 = stats_cal_2(billsum_train)
print("[Billsum] ratio of article/summary: ", np.array(bill_stats_src2)/np.array(bill_stats_tg2))