import torch
import numpy as np
import pandas as pd
from os.path import join


def load_latent_tensors(maps_path, group="validation", split=0):
    """
    Load latent embeding of a specifique group.
    """
    if group == "train" or group == "validation":
        df = pd.read_csv(
            join(maps_path, f"groups/{group}/split-{split}/data.tsv"),
            sep="\t"
        )
    else:
        df = pd.read_csv(
            join(maps_path, f"groups/{group}/data.tsv"),
            sep="\t"
        )

    subjects = df["participant_id"].tolist()
    sessions = df["session_id"].tolist()
    tensor_dir = join(
        maps_path,
        f"split-{split}",
        "best-loss",
        group,
        "latent_tensors"
    )
    latent_tensors_list = []

    for i in range(len(df)):
        sub, ses = subjects[i], sessions[i]
        path = join(tensor_dir, f"{sub}_{ses}_image-0_latent.pt")
        latent = torch.load(path).detach().numpy()
        latent_tensors_list.append(latent)

    return np.array(latent_tensors_list), subjects, sessions


def load_session_list(maps_path, group):
    """
    Load all sub/session from tsv
    """
    if group == "train" or group == "validation":
        tsv_file = path.join(maps_path, "groups", group, f"split-{split}", "data.tsv")
    else:
        tsv_file = path.join(maps_path, "groups", group, "data.tsv")
    df = pd.read_csv(tsv_file, sep="\t", usecols=["participant_id", "session_id"])
    return list(df.to_records(index=False))

