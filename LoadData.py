from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--ds", nargs=2, type=str, help="name of dataset, split")
args = parser.parse_args()


def rename_datasets(dataset):
    dataset = dataset.rename_column(dataset.column_names[0], "source")
    dataset = dataset.rename_column(dataset.column_names[1], "target")
    return dataset

def load_data(dataset_name: str, split_: str = "train"):
    version_dic = {"cnn_dailymail": "3.0.0", "xsum": "1.2.0","wiki_lingua": "english","scitldr_A": "Abstract","scitldr_F": "FullText","billsum": "3.0.0"}
    version = version_dic.get(dataset_name)

    if dataset_name == "scitldr_A" or dataset_name == "scitldr_F":
        dataset_name = "scitldr"

    dataset = load_dataset(dataset_name, version, split=split_)

    if dataset_name == "wiki_lingua":
        dataset = dataset.rename_column("article", "source")
    elif dataset_name == "scitldr":
        pass
    else:
        dataset = rename_datasets(dataset)

    return dataset
