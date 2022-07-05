from LoadData import get_args, read_js
from inspection import get_print_lens
from similarity import get_print_simi

if __name__ == "__main__":
    args = get_args()

    datasets = read_js("ds_name_list.json")

    if args.stats[0] == "length":

        get_print_lens(
            args.dataset[0], args.split[0], args.token_method[0], p=args.sample_propor
        )

    if args.stats[0] == "similarity":
        get_print_simi(args.dataset[0], args.split[0], p=args.sample_propor)
