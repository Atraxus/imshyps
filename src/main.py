import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

# Import from src
from analysis import analysis
from paramiterator import ParamIterator


def main():
    param_iter = ParamIterator()
    results = param_iter.iterate()
    analysis(results, param_iter.hyperparameters)


if __name__ == "__main__":
    main()
