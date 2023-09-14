import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


# This function takes in the results from the paramhandler and fits a random forest
# to the data. It then returns the importance scores for each hyperparameter.
# The results need to be a list of tuples (performance, hyperparameters) for each combination,
# where hyperparameters is a dictionary.
def fanova_importance_scores(results: list, hyperparameters: list):
    # defaults = [(param.name, param.default) for param in hyperparameters]

    # Create a list of lists for the hyperparameter values
    hp_values = []
    for hp in hyperparameters:
        if isinstance(results[0][1][hp.name], list):  # checking if value is a list
            hp_values.append(
                [str(tup[1][hp.name]) for tup in results]
            )  # Convert inner list to string
        else:
            hp_values.append([tup[1][hp.name] for tup in results])

    # Convert the list of lists to a 2D array
    hp_values = np.array(hp_values).T

    # Create a list of performances
    performances = [tup[0] for tup in results]

    # Encode the categorical variables
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    hp_values_encoded = encoder.fit_transform(hp_values)

    # Fit the random forest
    rf = RandomForestRegressor()
    rf.fit(hp_values_encoded, performances)

    # Print the importance scores
    print("Importance scores:")
    feature_names = encoder.get_feature_names_out()
    aggregated_importances = {hp.name: 0 for hp in hyperparameters}
    for i, name in enumerate(feature_names):
        hp_name = name.split("_")[0]
        aggregated_importances[
            hyperparameters[int(hp_name[1:])].name
        ] += rf.feature_importances_[i]
    for hp_name, importance in aggregated_importances.items():
        print(f"{hp_name}: {importance}")

    return aggregated_importances


# This function the variance for the results of a single hyperparameter and returns the importance score
# This takes the performances as a list without the corresponding hpvalue. Don´t pass a list tuples here.
def hp_variance(performances: list):
    variance = sum(
        [
            (performance - sum(performances) / len(performances)) ** 2
            for performance in performances
        ]
    ) / len(performances)
    return variance


def plot_hp(performances: list, hyperparameter: str):
    x = [tup[1] for tup in performances]
    y = [tup[0] for tup in performances]
    # Make sure plot is reset
    plt.clf()
    plt.plot(x, y, label=hyperparameter)
    plt.xlabel(hyperparameter)
    plt.ylabel("Performance Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/" + hyperparameter.replace(" ", "_") + ".png", dpi=300)


def filter_results_for_hyperparameter(results: list, param_name: str, defaults: list):
    # Convert default_values list of tuples to a dictionary for easier lookup
    default_dict = dict(defaults)

    # We will retain only those results where all parameters are at their default values,
    # except possibly for the one we're interested in.
    filtered_results = []
    for result in results:
        _, params = result
        if all(params[hp] == default_dict[hp] for hp in params if hp != param_name):
            filtered_results.append(result)
    return filtered_results


# Filter so that all results have default values for all hyperparameters except the one we are interested in
def get_performances(results: list, param_name: str, defaults: list):
    results = filter_results_for_hyperparameter(results, param_name, defaults)
    # Returns a list of tuples (performance, hyperparameter value) for a given hyperparameter
    performances = [(perf, hp[param_name]) for perf, hp in results]
    # Make sure its sorted by hyperparameter value
    performances.sort(key=lambda tup: tup[1])

    return performances


# This function takes the results from the paramhandler and plots the performance for each
# hyperparameter. It also calculates the importance score for each hyperparameter and prints
# it to the console. The results are saved in the plots folder.
# The results need to be a list of tuples (performance, hyperparameters) for each combination,
# where hyperparameters is a dictionary.
def analysis(
    file_name: str,
    results: list,
    hyperparameters: list,
    runtime: float = None,
):
    defaults = [(param.name, param.default) for param in hyperparameters]

    # Calculate fANOVA importance scores
    fanova_importances = fanova_importance_scores(results, hyperparameters)

    importance_scores = []
    for hp in hyperparameters:
        performances = get_performances(results, hp.name, defaults)
        plot_name = file_name + "_" + hp.name.replace(" ", "_")
        plot_hp(performances, plot_name)

        # Remove hp values
        just_performances = [tup[0] for tup in performances]
        variance_importance = hp_variance(just_performances)
        print(f"Variance Importance score for {hp.name}: {variance_importance}")
        print(f"fANOVA Importance score for {hp.name}: {fanova_importances[hp.name]}")
        importance_scores.append(
            (variance_importance, fanova_importances[hp.name], hp.name)
        )

    # Append importance scores to the log file in the plots folder
    with open("./log.txt", "a") as f:
        if runtime is not None:
            timeString = time.strftime("%H:%M:%S", time.gmtime(runtime))
        else:
            timeString = "N/A"
        num_samples = len(just_performances)

        # Get average performance for all runs
        avg_performance = sum(just_performances) / len(just_performances)

        f.write(
            file_name
            + "(num samples: "
            + str(num_samples)
            + ", time: "
            + timeString
            + ", avg performance: "
            + str(avg_performance)
            + ")\n"
        )
        for variance_score, fanova_score, hp in importance_scores:
            f.write(
                hp
                + ": Variance="
                + str(variance_score)
                + ", fANOVA="
                + str(fanova_score)
                + "\n"
            )
        f.write("\n")

    with open("./results_log.txt", "a") as f:
        f.write(file_name + ": " + str(avg_performance) + "\n")
        f.write("Results:\n")
        resultsString = ""
        for result in results:
            resultsString += "    " + str(result) + "\n"
        f.write(resultsString)
        f.write("\n")
