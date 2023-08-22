import time

import matplotlib.pyplot as plt


# This function applies the fANOVA algorithm to the results of a single hyperparameter and returns the importance score
# This takes the performances as a list without the corresponding hpvalue. Don´t pass a list tuples here.
def fanova(performances: list):
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
    model_name: str,
    results: list,
    hyperparameters: list,
    runtime: float = None,
):
    defaults = [(param.name, param.default) for param in hyperparameters]

    importance_scores = []
    for hp in hyperparameters:
        performances = get_performances(results, hp.name, defaults)
        plot_name = model_name + "_" + hp.name.replace(" ", "_")
        plot_hp(performances, plot_name)

        # Remove hp values
        just_performances = [tup[0] for tup in performances]
        print(f"Importance score for {hp.name}: {fanova(just_performances)}")
        importance_scores.append((fanova(just_performances), hp.name))

    # Append imporatance scores to the log file in the plots folder
    with open("./plots/log.txt", "a") as f:
        if runtime is not None:
            timeString = time.strftime("%H:%M:%S", time.gmtime(runtime))
        else:
            timeString = "N/A"
        num_samples = len(just_performances)
        f.write(
            model_name
            + "(num samples: "
            + str(num_samples)
            + ", time: "
            + timeString
            + ")\n"
        )
        for score, hp in importance_scores:
            f.write(hp + ": " + str(score) + "\n")
        f.write("\n")
