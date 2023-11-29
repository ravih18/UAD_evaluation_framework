from os import path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import nibabel as nib
from nilearn.image import resample_to_img

from utils.maps_reader import load_session_list
from utils.metrics import healthiness_score

## Parameters
maps_path = "/gpfswork/rech/krk/commun/anomdetect/VAE_evaluation/MAPS_VAE2"
split = 1

# Load region mask - il faut resampler le mask à la taille des images
# utiliser le meme resampling que pour le train
# Prendre la même dimension reduction que pour les tenseurs

def load_masks(pathology):
    """
    """
    caps_dir = "/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform"
    img_path = path.join(caps_dir, 
                     "subjects", 
                     "sub-ADNI002S0685", 
                     "ses-M48", 
                     "t1_linear", 
                     "sub-ADNI002S0685_ses-M48_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz")
    mni_path = "/gpfswork/rech/krk/usy14zi/vae_benchmark/data/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"

    mask_path = path.join(caps_dir, "masks", f"mask_hypo_{pathology}.nii")

    mask_nii = nib.load(mask_path)
    mask_nii = resample_to_img(mask_nii, nib.load(img_path), interpolation='nearest')
    pathology_mask = mask_nii.get_fdata()

    # Load MNI template mask
    mni_nii = nib.load(mni_path)
    mni_nii = resample_to_img(mni_nii, nib.load(img_path), interpolation='nearest')
    brain_mask = mni_nii.get_fdata()
    out_mask = brain_mask - pathology_mask
    
    return pathology_mask, out_mask


def compute_metrics(session, group, pathology_mask, reference_mask, columns):
    """
    """
    sub, ses = session[0], session[1]
    
    # Load all IO image
    gt_file = sub + "_" + ses + "_image-0_input.pt"
    gt_path = path.join(maps_path, f"split-{split}", "best-loss", "test_CN", "tensors", gt_file)
    gt_array = torch.load(gt_path).numpy()
    
    input_file = sub + "_" + ses + "_image-0_input.pt"
    input_path = path.join(maps_path, f"split-{split}", "best-loss", group, "tensors", input_file)
    input_array = torch.load(input_path).numpy()
    
    recon_file = sub + "_" + ses + "_image-0_output.pt"
    recon_path = path.join(maps_path, f"split-{split}", "best-loss", group, "tensors", recon_file)
    recon_array = torch.load(recon_path).detach().numpy()
    
    healthiness_gt = healthiness_score(gt_array, pathology_mask, reference_mask)
    healthiness_input = healthiness_score(input_array, pathology_mask, reference_mask)
    healthiness_recon = healthiness_score(recon_array, pathology_mask, reference_mask)
    
    group_label = ' '.join(group.upper().split('_')[2:])
    row1 = [sub, ses, group_label, "ground truth $x$", healthiness_gt]
    row2 = [sub, ses, group_label, "simulation $x'$", healthiness_input]
    row3 = [sub, ses, group_label, "reconstruction $\widehat{x'}$", healthiness_recon]

    return pd.DataFrame([row1, row2, row3], columns=columns.keys())


def make_healthiness_dataframe(groups, columns):
    """
    """
    results_df = pd.DataFrame(columns) 

    for group in groups:

        sessions_list = load_session_list(group)
        pathology_mask, out_mask = load_masks("ad")
        
        for session in sessions_list:
            row_df = compute_metrics(session, group, pathology_mask, out_mask, columns)
            results_df = pd.concat([results_df, row_df])
    return results_df


def heathiness_boxplot(groups, figure_name="healthiness"):
    """
    """
    columns = {
        "participant_id": pd.Series(dtype='str'),
        "session_id": pd.Series(dtype='str'),
        "group": pd.Series(dtype='str'),
        "measure":pd.Series(dtype='str'),
        "healthiness": pd.Series(dtype='float'),
    }

    results_df =  make_healthiness_dataframe(groups, columns)

    fig = plt.figure(figsize=(20, 8))
    ax = sns.boxplot(data=results_df, x="group", y="healthiness", hue="measure", orient='v')

    colors = sns.color_palette('tab20')
    handles = []
    for i in range(len(groups)+1):

        if i == 0:
            box_gt = ax.patches[0]
        else:
            box_gt = ax.patches[3*i+1]

        box_rec = ax.patches[3*i+2]
        
        rgb1 = to_rgb(colors[2*i])
        box_gt.set_facecolor(rgb1)
        
        rgb2 = to_rgb(colors[2*i+1])
        box_rec.set_facecolor(rgb2)

        if i!=0:
            handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb1, edgecolor='black'))
            handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb2, edgecolor='black'))

    plt.xlabel("Group", fontsize=20)
    plt.ylabel("Healthiness", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 0, 0, facecolor=to_rgb(colors[0]), edgecolor='black'),
            tuple(handles[::2]), 
            tuple(handles[1::2])
        ], 
        labels=["ground truth $x$", "simulation $x'$", "reconstruction $\widehat{x'}$"], 
        handlelength=8, 
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        fontsize=20
    )
    #ax.legend(loc='lower left', fontsize=20)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"plots/{figure_name}_boxplot.png")



if __name__ == "__main__":

    groups = [
        "test_hypo_ad_5",
        "test_hypo_ad_10",
        "test_hypo_ad_15",
        "test_hypo_ad_20",
        "test_hypo_ad_30",
        "test_hypo_ad_40",
        "test_hypo_ad_50",
        "test_hypo_ad_70",
    ]
    heathiness_boxplot(groups, 'healtiness_ad')

    groups = [
        "test_hypo_ad_30",
        "test_hypo_pca_30",
        "test_hypo_bvftd_30",
        "test_hypo_lvppa_30",
        "test_hypo_svppa_30",
        "test_hypo_nfvppa_30",
    ]
    heathiness_boxplot(groups, 'healtiness_ad')
