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


def get_unique_tuples(input_list):
    # TODO(Jannis): Currently taking the first tuple with a given key (greedy)
    unique_tuples = {}
    for tup in input_list:
        key = tup[1]
        if key not in unique_tuples:
            unique_tuples[key] = tup
    return list(unique_tuples.values())


def get_performances(results: list, param: str, defaults: list):
    # Returns a list of tuples (performance, hyperparameter value) for a given hyperparameter
    performances = [(perf, hp[param]) for perf, hp in results]
    unique_perfs = get_unique_tuples(performances)
    # Make sure its sorted by hyperparameter value
    unique_perfs.sort(key=lambda tup: tup[1])
    return unique_perfs


def analysis(model_name: str, results: list, hyperparameters: list):
    defaults = [(param.name, param.default) for param in hyperparameters]
    for hp in hyperparameters:
        accuracies = get_performances(results, hp.name, defaults)
        plot_name = model_name + "_" + hp.name.replace(" ", "_")
        plot_hp(accuracies, plot_name)
