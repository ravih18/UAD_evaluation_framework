import pandas as pd
import torch
from os.path import join
from sklearn.metrics import DistanceMetric
from pytorch_ssim import ssim3D
from joblib import Parallel, delayed
import pickle

from utils.maps_reader import load_latent_tensors
from utils.latent_space_distance import get_closest_img_part

### METADATA
caps_dir = "/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform"
maps_path = "/gpfswork/rech/krk/commun/anomdetect/VAE_evaluation/MAPS_VAE2"
split = 1

cpu = 10

### DEF FUNCTIONS
def read_img(participant):
    path = join(
        caps_dir,"subjects", participant[0], participant[1],
        "deeplearning_prepare_data/image_based/pet_linear/",
        f"{participant[0]}_{participant[1]}_trc-18FFDG_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.pt"
    )
    img = torch.load(path)
    return img

def compute_participant_metric(sub_ses):
    closest_participants, scores = get_closest_img_part(
        sub_ses, participants, dist, n=10, interval=1
    )
    img_a = read_img(sub_ses)
    MSE = []
    SSIM = []
    for neibourgh in closest_participants:
        img_b = read_img(neibourgh)
        MSE.append(100 * torch.nn.functional.mse_loss(img_a, img_b).item())
        SSIM.append(ssim3D(img_a.detach().numpy(), img_b.detach().numpy()).item())
    return [sub_ses, scores, MSE, SSIM]

## Load latent tensors
X_train, subject_train, session_train = load_latent_tensors(maps_path, "train", split)
participants = list(zip(subject_train, session_train))

# Pairwise distance
dist = DistanceMetric.get_metric('minkowski', p=10).pairwise(X_train)

## Parallelize the metric computation to save time
metrics = Parallel(n_jobs=cpu)(
    delayed(compute_participant_metric)(sub_ses) for sub_ses in participants
)

with open('data/metric_lmm.pkl', 'wb') as f:
    pickle.dump(metrics, f)

latent, mse, ssim, idx_col = [], [], [], []
for i in range(len(metrics)):
    latent.extend(metrics[i][1])
    mse.extend(metrics[i][2])
    ssim.extend(metrics[i][3])
    idx_col.extend([i]*n)

d = {
    'idx_col': idx_col,
    'latent': latent,
    'mse': mse,
    'ssim': ssim,
}
data = pd.DataFrame(data=d)

data.to_csv("data/latentvsimage.tsv", sep="\t")
