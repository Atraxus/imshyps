import json
import os

# Ensure experiments directory exists
if not os.path.exists("./configs/experiments/"):
    os.makedirs("./configs/experiments/")

# Define the base paths for the configs
BASE_PATHS = [
    "./configs/mlp.json",
    "./configs/esn.json",
    "./configs/lucas.json",
    "./configs/lenet5.json",
]

# Define the configurations to modify for each model
CONFIG_MODIFICATIONS = {
    "mlp.json": ["samples", "epochs"],
    "esn.json": ["samples"],
    "lucas.json": ["epochs"],
    "lenet5.json": ["samples", "epochs"],
}

# Iterate through the base paths and make modifications
for base_path in BASE_PATHS:
    with open(base_path, "r") as base_file:
        base_config = json.load(base_file)
        file_name = os.path.basename(base_path)

        # Determine which parameters to modify for this model
        params_to_modify = CONFIG_MODIFICATIONS[file_name]

        # Modify 'samples' if it's one of the parameters for this model
        if "samples" in params_to_modify:
            for value in range(5, 45, 5):
                new_config = base_config.copy()
                new_config["samples"] = value
                if "epochs" not in params_to_modify:
                    new_config["epochs"] = base_config["epochs"]  # Default value

                # Save the new configuration
                with open(
                    f"./configs/experiments/{file_name[:-5]}_samples_{value}.json", "w"
                ) as new_file:
                    json.dump(new_config, new_file, indent=4)

        # Modify 'epochs' if it's one of the parameters for this model
        if "epochs" in params_to_modify:
            for value in range(5, 45, 5):
                new_config = base_config.copy()
                new_config["epochs"] = value
                if "samples" not in params_to_modify:
                    new_config["samples"] = base_config["samples"]  # Default value

                # Save the new configuration
                with open(
                    f"./configs/experiments/{file_name[:-5]}_epochs_{value}.json", "w"
                ) as new_file:
                    json.dump(new_config, new_file, indent=4)
