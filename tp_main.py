if __name__ == "__main__":
    from sys import stdout

    from pathlib import Path

    import pandas as pd
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from datasets import MyArffDataset
    from k_mean import MyKMeans
    from agglomerative_clustering import MyAgglomerativeClustering

    out_dir = Path("./cluster_results/")
    out_dir.mkdir(exist_ok=True)

    TP_DATASET = ("xclara.arff", "twodiamonds.arff", "complex8.arff", "engytime.arff", "diamond9.arff")


    def save_image(name: str):
        plt.savefig(out_dir / f"{name}.svg")
        plt.clf()


    k_mean_res = []
    agglomerative_res = []

    for f in tqdm(TP_DATASET, desc="Processing datasets", file=stdout):
        dataset = MyArffDataset(f)
        dataset.plot_data()
        save_image(f"init.{f}")

        for k in range(2, 11):
            k_mean = MyKMeans(dataset, n_clusters=k)
            dataset.plot_labeled_data(k_mean.labels)
            save_image(f"k_mean.{f}.{k}")
            k_mean_res.append(k_mean)

        for threshold in range(1, 22, 2):
            model = MyAgglomerativeClustering(dataset, n_clusters=None, distance_threshold=threshold)
            dataset.plot_labeled_data(model.labels)
            save_image(f"agglomerative.{f}.t.{threshold}")
            agglomerative_res.append(model.values())

        for n_clusters in range(2, 11):
            model = MyAgglomerativeClustering(dataset, n_clusters=n_clusters, distance_threshold=None)
            dataset.plot_labeled_data(model.labels)
            save_image(f"agglomerative.{f}.n.{n_clusters}")
            agglomerative_res.append(model.values())

    table = pd.DataFrame([k_mean.values() for k_mean in k_mean_res], columns=MyKMeans.headers())
    table.to_csv(out_dir / "k_mean.csv", index=False)

    table = pd.DataFrame(agglomerative_res, columns=MyAgglomerativeClustering.headers())
    table.to_csv(out_dir / "agglomerative.csv", index=False)
