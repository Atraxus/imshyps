# This module is for testing this project

from analysis import analysis, fanova_importance_scores
from data import generate_random_results
from models import MLP as MODEL
from paramhandler import ParamHandler


def main():
    param_handler = ParamHandler(MODEL, MODEL.MODEL_HPARAMS, "configs/mlp.json")

    results = generate_random_results(
        "configs/mlp.json"
    )  # param_handler.run() dummy results

    fanova_importance_scores(results, param_handler.params)


if __name__ == "__main__":
    main()
