from pytorch_ssim import ssim_map_3D
from scipy.spatial.distance import minkowski


def mse_in_mask(X, Y, mask):
    return ((X - Y)**2).mean(where=mask.astype(bool))


def ssim_in_mask(X, Y, mask):
    ssim_map = ssim_map_3D(X, Y).numpy()
    return ssim_map.mean(where=mask.astype(bool))


def minkowski_distance(X, Y):
    return minkowski(X, Y, p=10)


def mean_in_mask(X, mask):
    return X.mean(where=mask.astype(bool))


def healthiness_score(X, mask, out_mask):
    return mean_in_mask(X, mask) / mean_in_mask(X, out_mask)


def create_mask(atlas, value):
    return (atlas==value).astype(int)


def anomaly_score(X, atlas, atlas_dict):
    score = {}
    # Iterate over the regions
    for region, value in atlas_dict.items():
        # make a mask of the region
        mask = create_mask(atlas, value)
        # compute the mean intensity within the region (region mean)
        rm = mean_in_mask(X, mask)
        score[region] = rm
    return score

