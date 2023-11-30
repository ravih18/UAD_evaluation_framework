import pandas as pd
from sklearn.metrics import DistanceMetric
import matplotlib.pyplot as plt
import seaborn as sns

from utils.maps_reader import load_latent_tensors
from utils.latent_space_distance import (
    get_closest_img_part,
    get_closest_img_same_part
)

def make_dataframe(maps_path, split):
    """
    """
    ## Load latent tensors
    X_train, subject_train, session_train = load_latent_tensors(maps_path, "train", split)
    participants = list(zip(subject_train, session_train))

    dist = DistanceMetric.get_metric('minkowski', p=10).pairwise(X_train)

    # Build the dataframe
    columns = {
        "participant_id": pd.Series(dtype='str'),
        "session_id": pd.Series(dtype='str'),
        "group": pd.Series(dtype='str'),
        "metric": pd.Series(dtype='float'),
    }
    df_distance = pd.DataFrame(columns)
    for sub_ses in participants:
        # Intra subject
        same_sub_distance = get_closest_img_same_part(sub_ses, participants, dist)
        if len(same_sub_distance)>1:
            row = [sub_ses[0], sub_ses[1], "intra-subject", sum(same_sub_distance[1:]) / len(same_sub_distance[1:])]
            row_df = pd.DataFrame([row], columns=columns.keys())
            df_distance = pd.concat([df_distance, row_df])
        # Inter subject
        _, same_sub_distance = get_closest_img_part(sub_ses, participants, dist)
        row = [sub_ses[0], sub_ses[1], "inter-subject", sum(same_sub_distance) / len(same_sub_distance)]
        row_df = pd.DataFrame([row], columns=columns.keys())
        df_distance = pd.concat([df_distance, row_df])
    return df_distance


def make_cp_boxplot(maps_path, split):
    """
    """
    df_distance = make_dataframe(maps_path, split)

    fig = plt.figure(figsize=(11, 4))
    sns.boxplot(
        df_distance,
        x= 'metric', y = 'group',
        orient='h', width=0.4
    )
    
    plt.xlabel("Minkowski distance (p=10)", fontsize=15)
    plt.ylabel("", rotation=0, fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='x')
    
    plt.tight_layout()
    plt.savefig("plots/boxplot_inter_vs_intra.png")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_path')
    parser.add_argument('-s', '--split', default=0)
    args = parser.parse_args()

    make_cp_boxplot(args.maps_path, args.split)
