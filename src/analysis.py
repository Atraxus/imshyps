import matplotlib.pyplot as plt


def fanova(results: list):
    # This function applies the fANOVA algorithm to the results of a single hyperparameter and returns the importance score
    val_losses = []
    for result in results:
        # Average validation loss over all runs
        val_losses.append(sum(result[3]) / len(result[3]))

    variance = sum([(val_loss - sum(val_losses) / len(val_losses))
                   ** 2 for val_loss in val_losses]) / len(val_losses)
    return variance


# TODO(Jannis): Epochs as parameter here are inelegant
def plot_val_loss(hp_results: list, hyperparameter: str, num_epochs: int = 1):
    # This function plots the validation loss over the epochs for each hyperparameter
    # As input it expects a list with hp_value - val_loss pairs
    # val_loss is a list of validation losses for each epoch
    hp_values = [hp_result[0] for hp_result in hp_results]
    epochs = []
    for i in range(num_epochs):
        epochs.append([hp_result[1][i] for hp_result in hp_results])

    # Make sure plot is reset
    plt.clf()
    for i, epoch in enumerate(epochs):
        plt.plot(hp_values, epoch, label="Epoch " + str(i + 1))
    plt.xlabel(hyperparameter)
    plt.ylabel("Validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/" + hyperparameter.replace(" ", "_") + ".png", dpi=300)


def analyze_results(hyperparameters: list, results: list):
    # This function analyzes the importance of the hyperparameters
    lrate_importance = fanova(results[0])
    bsize_importance = fanova(results[1])
    afun_importance = fanova(results[2])

    print("Learning rate importance score: " + str(lrate_importance))
    print("Batch size importance score: " + str(bsize_importance))
    print("Activation function importance score: " + str(afun_importance))


def analysis(results: list, hyperparameters: list):
    analyze_results(hyperparameters, results)
    # Plot lrate
    lrate_results = []
    for result in results[0]:
        lrate_results.append((result[0], result[3]))
    plot_val_loss(lrate_results, "Learning rate")

    # Plot bsize
    bsize_results = []
    for result in results[1]:
        bsize_results.append((result[1], result[3]))
    plot_val_loss(bsize_results, "Batch size")

    # Plot afun
    afun_results = []
    for result in results[2]:
        afun_results.append((result[2], result[3]))
    plot_val_loss(afun_results, "Activation function")
