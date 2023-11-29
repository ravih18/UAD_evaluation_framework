import os
from os import path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from tqdm.notebook import tqdm
from nilearn.image import resample_to_img, binarize_img
from nilearn.masking import apply_mask
from sklearn.preprocessing import binarize
from scipy.stats import mannwhitneyu, normaltest
from statannotations.Annotator import Annotator

from utils.maps_reader import load_session_list
from utils.metrics import anomaly_score

##
maps_path = "/gpfswork/rech/krk/commun/anomdetect/VAE_evaluation/MAPS_VAE2"
split = 1

def get_dataframe(group):

    sessions_list = load_session_list(group)

    # Load atlas
    atlas_gm = nib.load("data/AAL2/AAL2_mni_gm.nii").get_fdata()
    atlas_df = pd.read_csv("data/AAL2/AAL2_new_index.tsv", sep="\t")
    atlas_dict = dict(zip(atlas_df.Region, atlas_df.Value))
    regions = list(atlas_df.Region)

    columns = {
        "participant_id": pd.Series(dtype='str'),
        "session_id": pd.Series(dtype='str'),
        "image": pd.Series(dtype='str'),
        "region": pd.Series(dtype='str'),
        "metric": pd.Series(dtype='float'),
    }
    df = pd.DataFrame(columns)
    
    def compute_metrics(session, atlas, atlas_dict):
        sub, ses = session[0], session[1]

        # Load all IO image
        input_file = sub + "_" + ses + "_image-0_input.pt"
        input_path = path.join(maps_path, f"split-{split}", "best-loss", group, "tensors", input_file)
        input_array = torch.load(input_path).numpy()

        gt_file = sub + "_" + ses + "_image-0_input.pt"
        gt_path = path.join(maps_path, f"split-{split}", "best-loss", "test_CN", "tensors", gt_file)
        gt_array = torch.load(gt_path).numpy()
        
        recon_file = sub + "_" + ses + "_image-0_output.pt"
        recon_path = path.join(maps_path, f"split-{split}", "best-loss", group, "tensors", recon_file)
        recon_array = torch.load(recon_path).detach().numpy()

        score_inp = anomaly_score(input_array, atlas, atlas_dict)
        score_gt = anomaly_score(gt_array, atlas, atlas_dict)
        score_rec = anomaly_score(recon_array, atlas, atlas_dict)
        return score_inp, score_gt, score_rec

    for session in tqdm(sessions_list):
        score_inp, score_gt, score_rec = compute_metrics(session, atlas_gm, atlas_dict)
        for region in regions:
            row = pd.DataFrame([[session[0], session[1], "Simulated image", region, score_inp[region]]], columns=columns.keys())
            df = pd.concat([df, row])
            row = pd.DataFrame([[session[0], session[1], "Original image", region, score_gt[region]]], columns=columns.keys())
            df = pd.concat([df, row])
            row = pd.DataFrame([[session[0], session[1], "Network reconstruction", region, score_rec[region]]], columns=columns.keys())
            df = pd.concat([df, row])
    return df
    
def make_box_plot(df, group):
    
    regions = list(pd.read_csv("data/AAL2/AAL2_new_index.tsv", sep="\t").Region)
    regions.remove('Background')
    
    pairs = []
    for region in regions:
        pairs.append([(region, 'Simulated image'), (region, 'Network reconstruction')])
        pairs.append([(region, 'Original image'), (region, 'Simulated image')])
        pairs.append([(region, 'Original image'), (region, 'Network reconstruction')])
    
    hue_plot_params = {
        'data':      df,
        'x':         'region',
        'y':         'metric',
        "hue":       "image",
        "hue_order": ['Original image', 'Simulated image', 'Network reconstruction']
    }
    
    with sns.plotting_context('notebook', font_scale = 1.4):
        fig, ax = plt.subplots(1, 1, figsize=(25, 8))
        #plt.figure(figsize=(25, 6))

        ax = sns.boxplot(**hue_plot_params)
        
        # Add annotations
        annotator = Annotator(ax, pairs, **hue_plot_params)
        annotator.configure(test="Mann-Whitney", comparisons_correction="bonferroni")
        _, results = annotator.apply_and_annotate()
        
        plt.xticks(rotation=20)
        plt.xlabel("Regions of the brain", fontsize=20)
        plt.ylabel("Mean uptake", fontsize=20)
        ax.legend(loc='lower left', fontsize=20)
        plt.grid(axis='y')
        #plt.show()
        plt.tight_layout()
        plt.savefig(f"boxplot_{group}_stats.png")


if __name__ == "__main__":

    group = f"test_hypo_ad_30"
    df = get_dataframe(group)
    make_box_plot(df, group)
