from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
from numpy._typing import ArrayLike
from scipy.io import arff


class MyArffDataset:
    datasets_folder = Path("./datasets/src/main/resources/datasets/artificial/")

    def __init__(self, file: str):
        data, meta = arff.loadarff(open(self.datasets_folder / file, "r"))
        self.data = np.array([[x[0], x[1]] for x in data])
        self.meta = meta

    def plot_data(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], s=8)
        # plt.title(f"Données de {self.meta.name}")

    def plot_labeled_data(self, labels: ArrayLike | Sequence[ColorType] | ColorType):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=labels, s=8)
        # plt.title(f"Données labellisées de {self.meta.name}")


if __name__ == "__main__":
    dataset = MyArffDataset("xclara.arff")

    dataset.plot_data()
    plt.show()
