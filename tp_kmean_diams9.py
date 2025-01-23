import pandas as pd

if __name__ == "__main__":
    from sys import stdout

    from tqdm import tqdm
    from pathlib import Path

    from matplotlib import pyplot as plt

    from datasets import MyArffDataset
    from k_mean import MyKMeans


    def save_image(name: str):
        plt.savefig(out_dir / f"{name}.svg")
        plt.clf()

    d = {}

    out_dir = Path("./diamond_results/")
    out_dir.mkdir(exist_ok=True)
    datasets = MyArffDataset("diamond9.arff")
    for i in tqdm(range(100), desc="Processing datasets", file=stdout):
        k_mean = MyKMeans(datasets, n_clusters=9)
        datasets.plot_labeled_data(k_mean.labels)
        save_image(f"{k_mean.silhouette * 100:.2f}")
        if k_mean.silhouette not in d:
            d[k_mean.silhouette] = 1
        else:
            d[k_mean.silhouette] += 1


    table = pd.DataFrame(d.items(), columns=("Silhouette", "Nombre"))
    table.to_csv(out_dir / "result.csv", index=False)