#!/usr/bin/env python3

from pathlib import Path
import argparse

import listing

# Version of the format where I save the models, cases, etc. If in the future
# I change the format I can just change the string, so a new folder will be made
# and old things will be left ignored in old folder
DATA_VERSION = "v5"
CASES_PATH = Path(f"./cases/{DATA_VERSION}/")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def train(args):
        # Import now because loads tensorflow and I dont want to be done in
        # other cases because it is slow
        import train
        train.train(CASES_PATH)

    parser_train = subparsers.add_parser(
            "train",
            help=("Train case. It will create the model defined in the source "
                  "code, train it, and save everything in the \"cases\" folder: "
                  "the model, a json description, etc.")
        )
    parser_train.set_defaults(func=train)

    def eval_(args):
        # Import now because loads tensorflow and I dont want to be done in
        # other cases because it is slow
        import evaluate
        evaluate.evaluate(
                args.id, CASES_PATH,
                args.plot_history, args.history_ignore,
                args.examples
            )

    parser_eval = subparsers.add_parser(
            "eval",
            help=("Evaluate case. It will load the given model, evaluate it, "
                  "show plots, etc.")
        )
    parser_eval.add_argument("id",
            type=str,
            help="ID of case to evaluate",
        )
    parser_eval.add_argument("--plot-history", "-p",
            action="store_true",
            help="Plot history",
        )
    parser_eval.add_argument("--history-ignore", "-i",
            nargs="+",
            type=str,
            default=[],
            help="Metrics to ignore when showing history",
        )
    parser_eval.add_argument("--examples", "-e",
            action="store_true",
            help="Plot examples",
        )
    parser_eval.set_defaults(func=eval_)

    def list_(args):
        listing.listing(CASES_PATH, args.filter, args.verbose)

    parser_eval = subparsers.add_parser(
            "list",
            help=("List cases. It searchs all the saved cases/models and lists "
                  "them. Allows to filter results and select information to "
                  "show."))
    parser_eval.add_argument("--verbose", "-v",
            action="store_true",
            help="Show all information about case",
        )
    parser_eval.add_argument("--filter", "-f",
            type=str,
            nargs=2,
            help="Filter field of case description",
        )
    parser_eval.set_defaults(func=list_)

    args = parser.parse_args()
    # Call function corresponding to the selected subparser
    args.func(args)
