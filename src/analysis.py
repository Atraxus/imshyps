import matplotlib.pyplot as plt
from hyperparameter import HyperParameter


def fanova(results: list):
    # This function applies the fANOVA algorithm to the results of a single hyperparameter and returns the importance score
    performances = []
    for result in results:
        # Average validation loss over all runs
        performances.append(sum(result[3]) / len(result[3]))

    variance = sum([(performance - sum(performances) / len(performances))
                   ** 2 for performance in performances]) / len(performances)
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
        accuracy, params = result
        if all(params[hp] == default_dict[hp] for hp in params if hp != param_name):
            filtered_results.append(result)
    return filtered_results


def get_performances(results: list, param_name: str, defaults: list):
    # Filter so that all results have default values for all hyperparameters except the one we are interested in
    results = filter_results_for_hyperparameter(results, param_name, defaults)

    # Returns a list of tuples (performance, hyperparameter value) for a given hyperparameter
    performances = [(perf, hp[param_name]) for perf, hp in results]
    # Make sure its sorted by hyperparameter value
    performances.sort(key=lambda tup: tup[1])
    return performances


def analysis(model_name: str, results: list, hyperparameters: list):
    defaults = [(param.name, param.default) for param in hyperparameters]
    for hp in hyperparameters:
        accuracies = get_performances(results, hp.name, defaults)
        plot_name = model_name + "_" + hp.name.replace(" ", "_")
        plot_hp(accuracies, plot_name)
