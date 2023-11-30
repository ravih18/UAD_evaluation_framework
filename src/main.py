from latent_space import make_latent_space_plots
from closest_participant import make_cp_boxplot
from lmm import make_spaghetti_plots
from healthiness import make_heathiness_boxplot
from anomaly_score import make_anomaly_box_plot
from violin_plot import make_violin_plot
from reconstruction_figure import make_reconstruction_plot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('maps_path')
parser.add_argument('-s', '--split', default=0)
args = parser.parse_args()

maps_path = args.maps_path
split = args.split

test_sets_percentages = [
    "test_hypo_ad_5",
    "test_hypo_ad_10",
    "test_hypo_ad_15",
    "test_hypo_ad_20",
    "test_hypo_ad_30",
    "test_hypo_ad_40",
    "test_hypo_ad_50",
    "test_hypo_ad_70",
]
test_sets_pathologies = [
    "test_hypo_ad_30",
    "test_hypo_pca_30",
    "test_hypo_bvftd_30",
    "test_hypo_lvppa_30",
    "test_hypo_svppa_30",
    "test_hypo_nfvppa_30",
]

print("Latent Space analysis")
make_latent_space_plots(maps_path, split)
make_cp_boxplot(maps_path, split)

print("Linear mixed effect model")
make_spaghetti_plots(maps_path, split)

print("Healthiness")

make_heathiness_boxplot(maps_path, split, test_sets_percentages, 'healthiness_percentages')
make_heathiness_boxplot(args.maps_path, args.split, test_sets_pathologies, 'healthiness_pathologies')

print("Anomaly score")
make_anomaly_box_plot(maps_path, split, "test_hypo_ad_30")

print("Violin plot")
make_violin_plot(maps_path, split, ["Test_CN"].extend(test_sets_percentages))

print("Reconstruction")
sub = "sub-ADNI009S5147"
ses = "ses-M00"
make_reconstruction_plot(maps_path, split, sub, ses)
