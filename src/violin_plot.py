from os import path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from utils.maps_reader import load_session_list


def make_dataframe(maps_path, split, test_sets):
    """
    """
    columns = {
        "participant_id": pd.Series(dtype='str'),
        "session_id": pd.Series(dtype='str'),
        "Test set": pd.Series(dtype='int'),
        "Measure": pd.Series(dtype='str'),
        "Metric": pd.Series(dtype='float'),
    }
    results_df = pd.DataFrame(columns)

    for test_set in test_sets:
        # Load all sub/session from tsv
        sessions_list = load_session_list(maps_path, test_set)

        for session in sessions_list:
            sub, ses = session[0], session[1]
            if sub != "sub-ADNI067S0257":

                # Load all IO image
                gt_file = sub + "_" + ses + "_image-0_input.pt"
                gt_path = path.join(maps_path, f"split-{split}", "best-loss", "test_CN", "tensors", gt_file)
                gt_array = torch.load(gt_path).numpy()

                gt_recon_file = sub + "_" + ses + "_image-0_output.pt"
                gt_recon_path = path.join(maps_path, f"split-{split}", "best-loss", "test_CN", "tensors", gt_recon_file)

                input_file = sub + "_" + ses + "_image-0_input.pt"
                input_path = path.join(maps_path, f"split-{split}", "best-loss", test_set, "tensors", input_file)
                input_array = torch.load(input_path).numpy()

                recon_file = sub + "_" + ses + "_image-0_output.pt"
                recon_path = path.join(maps_path, f"split-{split}", "best-loss", test_set, "tensors", recon_file)
                recon_array = torch.load(recon_path).detach().numpy()

                # Compute MSE on whole image
                mse_gt = ((gt_array[0] - recon_array[0])**2).mean()
                mse_input = ((input_array[0] - recon_array[0])**2).mean()

                row = [sub, ses, test_set, "reconstruction $\widehat{x'}$ - ground truth $x$", mse_gt]
                row_df = pd.DataFrame([row], columns=columns.keys())
                results_df = pd.concat([results_df, row_df])
                row = [sub, ses, test_set, "reconstruction $\widehat{x'}$ - network input $x'$", mse_input]
                row_df = pd.DataFrame([row], columns=columns.keys())
                results_df = pd.concat([results_df, row_df])

    return results_df


def make_violin_plot(maps_path, split, test_sets):
    """
    """
    results_df = make_dataframe(maps_path, split, test_sets)

    mean_value = results_df[
        results_df["Test set"]==0
    ][
        results_df["Measure"]=="reconstruction $\widehat{x'}$ - ground truth $x$"
    ]["Metric"].mean()

    results_df.Metric *= (1/mean_value)

    with sns.plotting_context('notebook', font_scale = 1.4):
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        
        ax = sns.violinplot(
            data=results_df,
            x='Test set',
            y='Metric',
            hue="Measure",
            split=True,
            order=range(0,75,5),
            width=0.9,
            inner="quartile",
            linewidth=2.5,
        )
        
        ax.set_xlabel('Percentage of simulated hypometabolism', fontsize=24)
        ax.set_ylabel('MSE', fontsize=24)
        plt.legend(loc='upper left', fontsize=24)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 8, 10, 14], fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(axis='y')
        plt.tight_layout()

        plt.savefig("plots/violinplot_ad.png")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_path')
    parser.add_argument('-s', '--split', default=0)
    args = parser.parse_args()

    test_sets = [
        "test_CN",
        "test_hypo_ad_5",
        "test_hypo_ad_10",
        "test_hypo_ad_15",
        "test_hypo_ad_20",
        "test_hypo_ad_30",
        "test_hypo_ad_40",
        "test_hypo_ad_50",
        "test_hypo_ad_70",
    ]
    
    make_violin_plot(args.maps_path, args.split, test_sets)