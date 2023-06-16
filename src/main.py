# Import from src
from analysis import analysis
from paramhandler import ParamHandler
# To change the network used, change the following line to import other network as MODEL.
# You will also change the config file used
from models import MLP as MODEL
from data import generate_random_results

config_path = "configs/mlp.json"  # Change this to fit the model


def main():
    results = generate_random_results()
    # Paramhandler is initialized with the network class and the hyperparameters that the network needs.
    param_handler = ParamHandler(
        MODEL, MODEL.MODEL_HPARAMS, config_path)
    print("Will run for a total of " +
          str(param_handler.total_num_samples()) + " samples")
    print("It will use the following hyperparameters:" +
          str(MODEL.MODEL_HPARAMS))
    results = param_handler.run()
    model_name = MODEL.__name__
    analysis(model_name, results, param_handler.params)


if __name__ == "__main__":
    main()
