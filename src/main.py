import json
import os
import time

# Import from src
from analysis import analysis

# Import all models
from models import MLP, EchoStateNetwork, LeNet5, LucasModel
from paramhandler import ParamHandler

# Create a dictionary to map model names to classes
MODEL_MAPPING = {
    "MLP": MLP,
    "EchoStateNetwork": EchoStateNetwork,
    "LeNet5": LeNet5,
    "LucasModel": LucasModel,
}


def main():
    config_directory = "configs"
    config_files = [f for f in os.listdir(config_directory) if f.endswith(".json")]

    for config_file in config_files:
        config_path = os.path.join(config_directory, config_file)

        with open(config_path, "r") as f:
            config_data = json.load(f)
            model_name = config_data["model"]

        MODEL = MODEL_MAPPING.get(model_name)
        if MODEL is None:
            print(
                f"Model '{model_name}' not found for config {config_file}. Skipping..."
            )
            continue

        param_handler = ParamHandler(MODEL, MODEL.MODEL_HPARAMS, config_path)
        print(f"Running for config: {config_file}")
        print(f"Will run for a total of {param_handler.total_num_samples()} samples")
        print(f"It will use the following hyperparameters: {MODEL.MODEL_HPARAMS}")

        param_handler.load_data(test_size=0.2)

        start_time = time.time()
        results = param_handler.run()
        elapsed_time = time.time() - start_time

        # File name without .json extension
        file_name = config_file[:-5]
        analysis(file_name, results, param_handler.params, runtime=elapsed_time)


if __name__ == "__main__":
    main()
