# Import from src
from analysis import analysis

# ! To change the network used, change the following line to import other network as MODEL.
from models import EchoStateNetwork as MODEL
from paramhandler import ParamHandler


def main():
    config_path = "configs/esn.json"  # ! Change this to fit the model

    # Paramhandler is initialized with the network class and the hyperparameters that the network needs.
    param_handler = ParamHandler(MODEL, MODEL.MODEL_HPARAMS, config_path)
    print(
        "Will run for a total of " + str(param_handler.total_num_samples()) + " samples"
    )
    print("It will use the following hyperparameters:" + str(MODEL.MODEL_HPARAMS))

    # Load data
    input_path = "data/temp_europa_2015-2019.nc"
    target_path = "data/targets.csv"
    param_handler.load_data(input_path, target_path, test_size=0.2)

    # results = generate_random_results()
    results = param_handler.run()

    model_name = MODEL.__name__
    analysis(model_name, results, param_handler.params)


if __name__ == "__main__":
    main()
