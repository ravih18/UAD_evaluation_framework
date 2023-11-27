import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.maps_reader import load_latent_tensors

## Plotting function
def plot_embedding(data: list[dict], figure_name: str="embedding"):
    fig = plt.figure(figsize=(12,8))

    for d in data:
        x, y = d["emb"].T
        plt.scatter(x, y, c=d["color"], label=d["label"])
    plt.legend(loc=2, fontsize = 18)
    plt.xlim([-13, 13])
    plt.ylim([-8, 8])
    plt.tight_layout()
    plt.savefig(f"plots/{figure_name}.png")

## Parameters
maps_path = "/gpfswork/rech/krk/commun/anomdetect/VAE_evaluation/MAPS_VAE2"
split = 1

## Load latent tensors
X_train, _, _ = load_latent_tensors(maps_path, "train", split)
X_test_cn, _, _ = load_latent_tensors(maps_path, "test_CN", split)
X_hypo, _, _ = load_latent_tensors(maps_path, "test_hypo_ad_25", split)
X_test_ad, _, _ = load_latent_tensors(maps_path, "test_AD", split)

## Fit a PCA on train set and get PCA on other
pca = PCA(n_components=2)
pca.fit(X_train)

## Plot train set and CN test
train_dict = {
    "emb": pca.transform(X_train),
    "color": "skyblue",
    "label": "Train set",
}
test_cn_dict = {
    "emb": pca.transform(X_test_cn),
    "color": "g",
    "label": "Test CN",
}
test_hypo = {
    "emb": pca.transform(X_hypo),
    "color": "",
    "label": "Test hypo",
}
test_ad = {
    "emb": pca.transform(X_test_ad),
    "color": "mediumvioletred",
    "label": "Test AD",
}

plot_embedding([train_dict, test_cn_dict], figure_name="emb_train")
plot_embedding([train_dict, test_cn_dict, test_hypo], figure_name="emb_hypo")
plot_embedding([train_dict, test_ad], figure_name="emb_test")


## Percentage plot
percentage = [5, 10, 15, 20, 25, 30, 40, 50 , 70]
colors_list = ["peachpuff", "lightsalmon", "coral", "salmon", "tomato", "orangered", "red", "crimson"]
data = [train_dict, test_cn_dict]

for i, p in enumerate(percentage):
    X_hypo = load_latent_tensors(maps_path, f"test_hypo_ad_{p}", split)
    X_hypo_ad_dict = {
        "emb": pca.transform(X_hypo),
        "color": colors_list[i],
        "label": f"AD {p}",
    }
    data.append(X_hypo_ad_dict)

#plt.legend(loc=3, ncol=3, fontsize = 18)
plot_embedding(data, figure_name="emb_hypo_ad")

