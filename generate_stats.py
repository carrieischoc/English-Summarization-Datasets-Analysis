from LoadData import get_args
from inspection import get_print_lens
from similarity import get_print_simi

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

    if args.sts[0] == "len":

        get_print_lens(args.d[0], args.d[1], args.t[0], p=args.p)

    if args.sts[0] == "simi":
        get_print_simi(args.d[0], args.d[1], p=args.p)
