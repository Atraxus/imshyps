# This module is for testing this project

from data import generate_random_results
from analysis import analysis
from models import MLP as MODEL
from paramhandler import ParamHandler


def main():
    param_handler = ParamHandler(
        MODEL, MODEL.MODEL_HPARAMS, "configs/mlp.json")

    results = generate_random_results(
        "configs/mlp.json")  # param_handler.run() dummy results

    analysis(MODEL.__name__, results, param_handler.params)


if __name__ == "__main__":
    main()
