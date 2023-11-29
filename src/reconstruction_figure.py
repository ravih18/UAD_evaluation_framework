from os.path import join
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.transform import resize
from scipy import ndimage
from nilearn.image import resample_to_img

maps_path = "/gpfswork/rech/krk/commun/anomdetect/VAE_evaluation/MAPS_VAE2"
split = 1

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def get_mask_imgs(pathology="ad", percentage=30):
    img_path = join("data", "mni_resolution.nii.gz")
    mask_path = join("data", "masks", f"mask_hypo_{pathology}.nii")

    mask_nii = nib.load(mask_path)
    mask_nii = resample_to_img(mask_nii, nib.load(img_path), interpolation='nearest')
    mask = mask_nii.get_fdata()

    mask = percentage / 100 * mask
    mask = ndimage.gaussian_filter(mask, sigma=5)

    mask_imgs = [
        resize(np.rot90(mask[:, :, 90]), (160, 160)),
        resize(np.rot90(mask[95, :, :]), (160, 160)),
        resize(np.rot90(mask[:, 58, :]), (160, 160)),
    ]
    return mask_imgs


def get_images(sub, ses, pathology='ad', percentage=30):

    cn_tensor_dir = join(maps_path, f"split-{split}", "best-loss/test_CN/tensors")
    cn_in_file = join(cn_tensor_dir, f"{sub}_{ses}_image-0_input.pt")
    cn_out_file = join(cn_tensor_dir, f"{sub}_{ses}_image-0_output.pt")

    cn_hypo_tensor_dir = join(
        maps_path, f"split-{split}",
        f"best-loss/test_hypo_{pathology}_{percentage}/tensors"
    )
    cn_hypo_in_file = join(cn_hypo_tensor_dir, f"{sub}_{ses}_image-0_input.pt")
    cn_hypo_out_file = join(cn_hypo_tensor_dir, f"{sub}_{ses}_image-0_output.pt")

    cn_in = torch.load(cn_in_file).detach().numpy()[0]
    cn_out = torch.load(cn_out_file).detach().numpy()[0]

    cn_hypo_in = torch.load(cn_hypo_in_file).detach().numpy()[0]
    cn_hypo_out = torch.load(cn_hypo_out_file).detach().numpy()[0]
    
    def extract_slices(X, Y):
        # Sagital setting
        X_sag = resize(np.rot90(X[95, :, :]), (160, 160))
        Y_sag = resize(np.rot90(Y[95, :, :]), (160, 160))
        
        # Coronal setting
        X_cor = resize(np.rot90(X[:, 58, :]), (160, 160))
        Y_cor = resize(np.rot90(Y[:, 58, :]), (160, 160))
        
        # Axial setting
        X_ax = resize(np.rot90(X[:, :, 90]), (160, 160))
        Y_ax = resize(np.rot90(Y[:, :, 90]), (160, 160))

        return [X_ax, Y_ax, X_ax-Y_ax, X_sag, Y_sag, X_sag-Y_sag, X_cor, Y_cor, X_cor-Y_cor]

    cn_img = extract_slices(cn_in, cn_out)
    cn_hypo_img = extract_slices(cn_hypo_in, cn_hypo_out)
    
    return [cn_img, cn_hypo_img]

def plot_reconstruction(sub, ses, pathology="ad", percentage=30):
    """
    """
    images =  get_images(sub, ses, pathology, percentage)
    mask_imgs = get_mask_imgs(pathology, percentage)

    labels_x = [
        "Input $x$",
        "Reconstruction $\hat{x}$",
        "Difference $x-\hat{x}$"
    ] * 3
    labels_y = [
        "Cognitively normal\nsubject",
        f"{pathology.upper()} {percentage}%\nhypometabolism",
        "Ground truth\nanomaly mask",
    ]

    fig_rows = 3
    fig_cols = 9
    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(30, 8.4),
        gridspec_kw={'wspace': 0,
                    'hspace': 0}
    )
    for i in range(fig_rows):
        for j in range(fig_cols):
            # Remove axis ticks
            axes[i][j].get_xaxis().set_ticks([])
            axes[i][j].get_yaxis().set_ticks([])
            
            if (j+1)%3==0:
                cmap = 'seismic'
                norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)
            else:
                cmap = 'nipy_spectral'
                norm = mpl.colors.Normalize(vmin=0, vmax=1)

            if i==2:
                if (j+1)%3==0:
                    axes[i][j].imshow(-1 * mask_imgs[((j+1)//3)-1], cmap=cmap, norm=norm)
                else:
                    axes[i][j].imshow(np.ones((160, 160)), cmap='Greys')
                    #axes[i][j].axis('off') # also removes the label
                    # make xaxis invisibel
                    axes[i][j].xaxis.set_visible(False)
                    # make spines (the box) invisible
                    plt.setp(axes[i][j].spines.values(), visible=False)
                    # remove ticks and labels for the left axis
                    axes[i][j].tick_params(left=False, labelleft=False)
                    #remove background patch (only needed for non-white background)
                    axes[i][j].patch.set_visible(False)
            else:
                axes[i][j].imshow(images[i][j], cmap=cmap, norm=norm)

            if i==0:
                axes[i][j].set_xlabel(labels_x[j], fontsize=18)
                axes[i][j].xaxis.set_label_position('top')
            if j==0:
                axes[i][j].set_ylabel(labels_y[i], fontsize=18)

    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], location='right', shrink=0.9, aspect=20, pad=0.01)
    plt.colorbar(mpl.cm.ScalarMappable(norm=MidpointNormalize(vmin=-1, vmax=1, midpoint=0), cmap='seismic'), cax=cax, **kw)
    #plt.tight_layout()
    plt.savefig(f"plots/reconstruction_{sub}_{ses}.png")

if __name__ == "__main__":

    sub = "sub-ADNI009S5147"
    ses = "ses-M00"
    plot_reconstruction(sub, ses)
