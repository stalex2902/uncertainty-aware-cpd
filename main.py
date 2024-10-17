import argparse
import warnings

from scripts.evaluate_ensemble import evaluate_all_ensembles

warnings.filterwarnings("ignore")


def get_parser():
    """Parse command line arguments for main.py"""

    parser = argparse.ArgumentParser(description="Evaluate ensemble")
    parser.add_argument(
        "--experiments_name",
        type=str,
        required=True,
        help="name of sdataset",
        choices=[
            "synthetic_1D",
            "synthetic_100D",
            "mnist",
            "human_activity",
            "explosion",
            "road_accidents",
            "yahoo",
        ],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type",
        choices=["seq2seq", "tscp"],
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help="Loss type for seq2seq model",
        choices=["indid", "bce"],
    )

    parser.add_argument(
        "--is_ensemble",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If true, evaluate ensemble, else Bayes model.",
    )

    parser.add_argument(
        "--n_models", type=int, default=10, help="Number of models in ensemble"
    )
    parser.add_argument(
        "-cal",
        "--calibrate",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="If True, calibrate the models using Beta calibration",
    )

    parser.add_argument(
        "-tn", "--threshold_number", type=int, default=300, help="threshold number"
    )

    parser.add_argument(
        "--eval_comp",
        "--evaluation_components",
        type=str,
        nargs="*",
        default=["mean", "min", "max", "median", "distances", "cusums"],
        help="What types of procedures to evaluate",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # boolean
    parser.add_argument(
        "--verbose",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="If true, print the metrics to the console.",
    )
    parser.add_argument(
        "--save_df",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="If true, save the dataframe with results.",
    )

    return parser


def main(args) -> None:
    """Hydra entrypoint.

    Args:
    ----
        cfg (DictConfig): hydra config.
    """
    (
        experiments_name,
        model_type,
        loss_type,
        is_ensemble,
        n_models,
        calibrate,
        threshold_number,
        evaluation_components,
        seed,
        verbose,
        save_df,
    ) = args.values()
    _ = evaluate_all_ensembles(
        experiments_name=experiments_name,
        model_type=model_type,
        loss_type=loss_type,
        is_ensemble=is_ensemble,
        n_models=n_models,
        calibrate=calibrate,
        threshold_number=threshold_number,
        evaluation_components=evaluation_components,
        seed=seed,
        verbose=verbose,
        save_df=save_df,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = dict(vars(parser.parse_args()))

    main(args)
