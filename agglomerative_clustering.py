import time

import pandas as pd
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

from datasets import MyArffDataset


class MyAgglomerativeClustering:
    def __init__(self, dataset: MyArffDataset, n_clusters: int | None = 2, distance_threshold: float | None = None):
        t1 = time.time()
        self._inner = cluster.AgglomerativeClustering(
            distance_threshold=distance_threshold,
            linkage='single',
            n_clusters=n_clusters)
        self._inner.fit(dataset.data)
        t2 = time.time()
        self.meta = dataset.meta
        self.time = (t2 - t1) * 1000
        if len(set(self._inner.labels_)) > 1:
            self.silhouette = silhouette_score(dataset.data, self._inner.labels_, metric='euclidean')
        else:
            self.silhouette = None

    @property
    def labels(self):
        return self._inner.labels_

    @property
    def n_clusters(self):
        return self._inner.n_clusters_

    @property
    def n_leaves(self):
        return self._inner.n_leaves_

    def log(self):
        print("Algorithme :     AgglomerativeClustering")
        print(f"Nom du dataset : {self.meta.name:>23}")
        print(f"Nombre de clusters : {self.n_clusters:16}")
        print(f"Nombre de feuilles : {self.n_leaves:16}")
        print(f"Temps d'exécution : {self.time:17.2f} ms")
        print(f"Score de silhouette : {self.silhouette:15.5f}")
        print("")

    @staticmethod
    def headers():
        return ["Nom du dataset", "Nombre de clusters", "Nombre de feuilles", "Temps d'exécution",
                "Score de silhouette"]

    def values(self):
        return [self.meta.name, self.n_clusters, self.n_leaves, self.time, self.silhouette]

    @classmethod
    def range(cls, dataset: MyArffDataset, range_thresholds: range = None, range_clusters: range = None):
        results = []
        if range_thresholds is not None:
            for threshold in range_thresholds:
                model = cls(dataset, n_clusters=None, distance_threshold=threshold)
                results.append(model.values())
        elif range_clusters is not None:
            for n_clusters in range_clusters:
                model = cls(dataset, n_clusters=n_clusters, distance_threshold=None)
                results.append(model.values())
        return pd.DataFrame(results, columns=cls.headers())


if __name__ == "__main__":
    from sys import stdout
    from pathlib import Path
    from tqdm import tqdm

    out_dir = Path("./cluster_results/")
    out_dir.mkdir(exist_ok=True)

    TP_DATASET = ("xclara.arff", "twodiamonds.arff", "complex8.arff", "engytime.arff", "diamond9.arff")

    for f in tqdm(TP_DATASET, desc="Processing datasets", file=stdout):
        dataset = MyArffDataset(f)

        # Analysis with range of distance thresholds
        table_thresholds = MyAgglomerativeClustering.range(dataset, range_thresholds=range(1, 11))
        table_thresholds.to_csv(out_dir / f"agglomerative_clustering_thresholds.{f}.csv", index=False)

        # Analysis with range of cluster numbers
        table_clusters = MyAgglomerativeClustering.range(dataset, range_clusters=range(2, 11))
        table_clusters.to_csv(out_dir / f"agglomerative_clustering_clusters.{f}.csv", index=False)

        model = MyAgglomerativeClustering(dataset, n_clusters=None, distance_threshold=10)
        model.log()
