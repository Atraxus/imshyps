# Import from src
from analysis import analysis
from paramhandler import ParamHandler
from network import MLP

def main():
    param_handler = ParamHandler()
    print("Will run for a total of " + str(param_handler.total_num_samples()) + " samples")
    print("It will use the following hyperparameters:" + str(MLP.MODEL_HPARAMS)) # TODO(Jannis): Modularize
    results = param_handler.run()
    analysis(results, param_handler.params)


if __name__ == "__main__":
    main()
