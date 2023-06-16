# Import from src
from analysis import analysis
from paramhandler import ParamHandler
# To change the network used, change the following line to import other network as MODEL.
from models import RandomForest as MODEL


def main():
    # Paramhandler is initialized with the network class and the hyperparameters that the network needs.
    param_handler = ParamHandler(
        MODEL, MODEL.MODEL_HPARAMS, "configs/random_forest.json")
    print("Will run for a total of " +
          str(param_handler.total_num_samples()) + " samples")
    print("It will use the following hyperparameters:" +
          str(MODEL.MODEL_HPARAMS))
    results = param_handler.run()
    model_name = MODEL.__name__
    analysis(model_name, results, param_handler.params)


if __name__ == "__main__":
    main()
