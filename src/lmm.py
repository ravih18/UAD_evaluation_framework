"""
This require the script utils/latent_space_lmm.py to run before.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from random import choice
import statsmodels.formula.api as smf

def plot_lmm(X, Y, mdf, metric="mse", colors=['teal', 'skyblue', 'royalblue']):
    fig = plt.figure(figsize=(12,10))
    for i in range(len(X)):
        color = choice(colors)
        plt.plot(X[i], Y[i], color, alpha = .5)

    lmem_params = mdf.params
    lmm_a = lmem_params.latent
    lmm_b = lmem_params.Intercept
    x_axes = np.linspace(2.5, 6, 100)
    y_axes = lmm_a * x_axes + lmm_b
    plt.plot(x_axes, y_axes, '--k', label=f"Linear mixed effect model: {lmm_a:.3f}*X + {lmm_b:.3f}")

    #plt.xlim([8, 14])

    plt.xlabel("Distance in latent space", fontsize=18)
    plt.ylabel("Structural similarity in image space", fontsize=18)
    plt.title("")

    plt.legend(prop={'size': 20})
    plt.grid()
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"plots/{metric}vslatent_lmm.png")


def make_spaghetti_plots():
    """
    """
    with open("data/metric_lmm.pkl", 'rb') as f:
        metrics = pickle.load(f)
    X_mse, Y_mse, X_ssim, Y_ssim = [], [], [], []
    for val in metrics:
        X_mse.append(val[1])
        Y_mse.append(val[2])
        X_ssim.append(val[1])
        Y_ssim.append(val[3])

    df = pd.read_csv("data/latentvsimage.tsv", sep="\t")

    md = smf.mixedlm("mse ~ latent", df, groups=df["idx_col"])
    mdf = md.fit()
    plot_lmm(X_mse, Y_mse, mdf, "mse", ['teal', 'skyblue', 'royalblue'])


    md = smf.mixedlm("ssim ~ latent", df, groups=df["idx_col"])
    mdf = md.fit()
    plot_lmm(X_ssim, Y_ssim, mdf, "ssim", ['tomato', 'salmon', 'lightsalmon'])


if __name__ == "__main__":

    make_spaghetti_plots()
