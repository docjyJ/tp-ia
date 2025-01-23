import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_score

from datasets import MyArffDataset


class MyKMeans:
    def __init__(self, dataset: MyArffDataset, n_clusters: int = 8):
        self._inner = cluster.KMeans(n_clusters=n_clusters, init="k-means++")
        self.meta = dataset.meta
        t1 = time.time()
        self._inner.fit(dataset.data)
        t2 = time.time()
        self.time = (t2 - t1) * 1000
        self.silhouette = silhouette_score(dataset.data, self._inner.labels_, metric='euclidean')

    @property
    def labels(self):
        return self._inner.labels_

    @property
    def n_clusters(self):
        return self._inner.n_clusters

    @property
    def n_iter(self):
        return self._inner.n_iter_

    def log(self):
        print("Algorithm :                    KMeans")
        print(f"Nom du dataset : {self.meta.name:>23}")
        print(f"Nombre de clusters : {self.n_clusters:16}")
        print(f"Nombre d'itérations : {self.n_iter:15}")
        print(f"Temps d'exécution : {self.time:17.2f} ms")
        print(f"Score de silhouette : {self.silhouette:15.5f}")
        # print(f"Score de Davies-Bouldin : {self.davies_bouldin:14.5f}")
        # print(f"Score de Calinski-Harabasz : {self.calinski_harabasz:11.5f}")
        print("")

    @staticmethod
    def headers():
        return ["Algorithm", "Nom du dataset", "Nombre de clusters", "Nombre d'itérations", "Temps d'exécution",
                "Score de silhouette"]

    def values(self):
        return ["KMeans", self.meta.name, self.n_clusters, self.n_iter, self.time, self.silhouette]

    @classmethod
    def range(cls, dataset: MyArffDataset, range_clusters: range = range(2, 11)):
        df = pd.DataFrame([cls(dataset, n_clusters=k).values() for k in range_clusters], columns=cls.headers())
        return df


if __name__ == "__main__":
    datasets = MyArffDataset("xclara.arff")

    k_means = MyKMeans(datasets, n_clusters=3)
    k_means.log()

    datasets.plot_labeled_data(k_means.labels)
    plt.show()
