from typing import List
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import median
import numpy as np
from LoadData import get_args
from inspection import get_lens
from similarity import get_simi

sns.set_theme(style="white", palette="pastel")
sns.despine(left=True)
paper_rc = {"lines.linewidth": 0.5, "lines.markersize": 15}
sns.set_context("paper", rc=paper_rc)


def make_1df(ds: str, fe: str, cls: str, sts: List) -> pd.DataFrame:
    df = pd.DataFrame()
    df[fe] = sts
    df = df.explode(fe, ignore_index=True)
    df["dataset"] = ds
    df["class"] = cls

    return df


def make_dfs(
    cls_sts: List, cls_names: List[str], ds_names: List[str], fe: str
) -> pd.DataFrame:
    df_list = []
    cls_n = len(cls_names)
    ds_n = len(ds_names)

    for i in range(ds_n):
        for j in range(cls_n):
            df_ = make_1df(ds_names[i], fe, cls_names[j], cls_sts[i * cls_n + j])
            df_list.append(df_)

    df = pd.concat(df_list, axis=0)
    df = df.reset_index(drop=True)

    return df


# Draw a nested violinplot and split the violins for easier comparison
def draw_violin(
    df: pd.DataFrame,
    name: str,
    y_: str = "length",
    hue_: str = "class",
    savefig=True,
    showm=True,
):
    _, ax = plt.subplots()

    sns.violinplot(
        data=df,
        x="dataset",
        y=y_,
        hue=hue_,
        showmeans=True,
        ax=ax,
        scale="count",
        cut=0,
        inner="quartile",
        dodge=0.6,
        palette="YlGnBu_d",
    )

    if showm:
        if y_ == "length":
            sns.pointplot(
                data=df,
                x="dataset",
                y=y_,
                hue=hue_,
                dodge=True,
                capsize=0.15,
                ci="sd",
                join=False,
                ax=ax,
                scale=1.2,
                palette={"ref": "#76EE00", "sum": "#76EE00"},
            )
            sns.pointplot(
                data=df,
                x="dataset",
                y=y_,
                hue=hue_,
                dodge=True,
                capsize=0.2,
                ci=0,
                join=False,
                estimator=median,
                ax=ax,
                scale=0,
                palette={"ref": "#FF6103", "sum": "#FF6103"},
            )
        if y_ == "similarity":
            sns.pointplot(
                data=df,
                x="dataset",
                y=y_,
                hue=hue_,
                dodge=0.6,
                capsize=0.15,
                ci="sd",
                join=False,
                ax=ax,
                scale=1.2,
                palette={"mean": "#76EE00", "max": "#76EE00", "min": "#76EE00"},
            )
            sns.pointplot(
                data=df,
                x="dataset",
                y=y_,
                hue=hue_,
                dodge=0.6,
                capsize=0.2,
                ci=0,
                join=False,
                estimator=median,
                ax=ax,
                scale=0,
                palette={"mean": "#FF6103", "max": "#FF6103", "min": "#FF6103"},
            )
            # sns.boxplot(data=df, x="dataset", y=y_, hue=hue_,showbox=False,showmeans=True,meanline=True,
            # showcaps=False, ax=ax,showfliers=False,zorder=10,whiskerprops={'visible': False},meanprops={'color': 'yellow', 'ls': '-', 'lw': 1},
            # medianprops={'color': 'red', 'ls': '-', 'lw': 1})

    handles, labels = ax.get_legend_handles_labels()
    if y_ == "length":
        labels_n = 1
    if y_ == "similarity":
        labels_n = 3

    ax.legend(
        handles[0:labels_n],
        labels[0:labels_n],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )

    if savefig:
        plt.savefig(f"plots/{name}_violin.png", bbox_inches="tight")


def draw_strip(
    df: pd.DataFrame,
    name: str,
    logscale=True,
    y_: str = "length",
    hue_: str = "class",
    savefig=True,
    showm=True,
):
    _, ax = plt.subplots()

    sns.stripplot(
        data=df,
        x="dataset",
        y=y_,
        hue=hue_,
        dodge=True,
        size=2,
        jitter=True,
        ax=ax,
        palette="YlGnBu_d",
    )
    # sns.boxplot(data=df, x="dataset", y=y_, hue=hue_,showbox=False,showmeans=True,meanline=True,
    #         showcaps=False, ax=ax,showfliers=False,zorder=10,
    #         whiskerprops={'visible': False},meanprops={'color': 'k', 'ls': '-', 'lw': 2},
    #         medianprops={'color': 'red', 'ls': '-', 'lw': 2})

    if showm:
        sns.pointplot(
            data=df,
            x="dataset",
            y=y_,
            hue=hue_,
            capsize=0.15,
            ci="sd",
            join=False,
            ax=ax,
            scale=1.2,
            palette={"ref": "k", "sum": "k"},
        )
        sns.pointplot(
            data=df,
            x="dataset",
            y=y_,
            hue=hue_,
            capsize=0.2,
            ci=0,
            join=False,
            estimator=median,
            ax=ax,
            scale=0,
            palette={"ref": "#FF6103", "sum": "#FF6103"},
        )

    if logscale:
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )

    if savefig:
        if logscale:
            plt.savefig(f"plots/{name}_log_strip.png", bbox_inches="tight")
        else:
            plt.savefig(f"plots/{name}_strip.png", bbox_inches="tight")


def draw_bar(df: pd.DataFrame, logscale=True, y_: str = "length", hue_: str = "class"):
    ax = sns.barplot(
        data=df,
        x="dataset",
        y=y_,
        hue=hue_,
        linewidth=0.5,
        ci="sd",
        capsize=0.1,
        palette="YlGnBu_d",
    )

    if logscale:
        ax.set_yscale("log")

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def plot_pos(pos: List, name: str, savefig=True):

    sns.histplot(
        pos,
        stat="probability",
        kde=True,
        binrange=(0, 1),
        color="green",
        line_kws=dict(lw=1.5),
    )
    plt.title(name)
    if savefig:
        plt.savefig(f"plots/{name}_pos.png")


def compute_show_ratio(sts: List, name: str, savefig=True):
    compression_ratio = sts[0] / sts[1]
    bw = np.ceil(np.max(compression_ratio) / 25)
    sns.histplot(compression_ratio, color="green", binwidth=bw)
    plt.axvline(x=np.mean(compression_ratio), label="mean", color="red")
    plt.axvline(x=np.max(compression_ratio), label="mean", color="green")
    plt.axvline(x=np.median(compression_ratio), label="median", color="k")
    plt.axvline(x=np.std(compression_ratio), label="std", color="gray", linestyle="--")
    plt.title(name)
    if savefig:
        plt.savefig(f"plots/{name}_cpRatio.png")


if __name__ == "__main__":
    args = get_args()

    datasets = [
        "cnn_dailymail",
        "wiki_lingua",
        "xsum",
        "scitldr_A",
        "scitldr_F",
        "billsum",
    ]
    cls_sts = []

    if args.sts[0] == "len":
        cls_names = ["ref", "sum"]

        for dataset in datasets:
            sts = get_lens(dataset, args.ds[1], args.tm[0])
            cls_sts.append(sts.src.lens)
            cls_sts.append(sts.tg.lens)

            # Histogram of compression ratio with vertical lines (mean - red, max - green, median - black, std - dashed gray)
            compute_show_ratio([sts.src.lens, sts.tg.lens], args.tm[0] + "_len")

        df = make_dfs(cls_sts, cls_names, datasets, "length")

        # draw all datasets in one of (w/o logscale) stripplot marked with mean and std (black), median (orange).
        draw_strip(df, args.tm[0] + "_len", logscale=False, showm=False)
        draw_strip(df, args.tm[0] + "_len")

        # draw violin plot of reference/summary marked with mean and std (green), median (orange).
        draw_violin(df[df["class"] == "ref"], args.tm[0] + "_len_ref")
        draw_violin(df[df["class"] == "sum"], args.tm[0] + "_len_sum")

    if args.sts[0] == "simi":
        cls_names = ["mean", "max", "min"]

        for dataset in datasets:
            sts = get_simi(dataset, args.ds[1])
            cls_sts.append(sts.mean)
            cls_sts.append(sts.max)
            cls_sts.append(sts.min)

            # Plot of distribution of relative position of most similar sentence in article
            plot_pos(sts.pos, dataset)

        df = make_dfs(cls_sts, cls_names, datasets, "similarity")

        # draw violin plot of distribution of mean, max, min of similarity, with mean and std (green), median (red)
        draw_violin(df, "simi", y_="similarity")
