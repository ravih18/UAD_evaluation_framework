# Evaluation of pseudo-healthy image reconstruction for anomaly detection in brain FDG PET

## Overview

This repository provides the source code and data necessary to run the evaluation of a deep generative model trained to reconstruct pseudo-healthy images for unsupervised anomaly detection on FDG PET.

The model has been trained using the (ClinicaDL)[https://clinicadl.readthedocs.io/en/latest/] open source software, and using the (ADNI dataset)[https://adni.loni.usc.edu/].

The method is described in the following preprint (Evaluation of pseudo-healthy image reconstruction for anomaly detection with deep generative models: Application to brain FDG PET)[https://hal.science/hal-04315738] (currently under revision).

If you use any ressources from this repository, please cite us:
```bibtex
@unpublished{hassanaly2023evaluation,
  TITLE = {{Evaluation of pseudo-healthy image reconstruction for anomaly detection with deep generative models: Application to brain FDG PET}},
  AUTHOR = {Hassanaly, Ravi and Brianceau, Camille and Solal, Ma{\"e}lys and Colliot, Olivier and Burgos, Ninon},
  NOTE = {working paper or preprint},
  YEAR = {2023},
}
```

## Requirement

### Environment

To come

### Data

In order to improve reproducibility, the ADNI dataset have been converted to [BIDS]() and preprocessed using Clinica open source software.

Once we obtain a BIDS of ADNI using the [`adni-to-bids`](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Converters/ADNI2BIDS/) command, we run the [`pet-linear`](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/PET_Linear/) command on the dataset:
```
clinica run pet-linear $BIDS_DIRECTORY $CAPS_DIRECTORY 18FFDG cerebellumPons2
```
The outputs are stored in a CAPS dataset at $CAPS_DIRECTORY.

All the details on the data selection are in the Appendix of the article.

### Model training

The model has been trained using ClinicaDL and [Pythae](https://pythae.readthedocs.io/en/latest/) using the following command. 

```
clinicadl train pythae $CAPS_DIRECTORY $PREPROCESSING_JSON $TSV_DIR $MAPS_DIRECTORY -c $CONFIG_FILE
```
See [ClinicaDL documentation](https://clinicadl.readthedocs.io/en/latest/) for more information on how to run this command. A script to run this command on a cluster is available, run it using the following command:
```
sbatch clinicadl_scripts/run_train.sh
```
after specifying in the file the different path to required files and folders.

A config file with the used parameters are available at in config/config_vae.toml

Outputs are stored in a MAPS directory at $MAPS_DIRECTORY.

### Hypometabolism simulation

We then need to build the different simulated dataset used for our evaluation.
For that we run following ClinicaDL command with the different parameters of `$PATHOLOGY` and `$PERCENTAGE`:
```
clinicadl generate hypometabolic $CAPS_DIRECTORY $GENERATED_CAPS_DIRECTORY --pathology $PATHOLOGY --anomaly_degree $PERCENTAGE
```

All the different test set build for the experiments are detailed in the article.

### Run inference on the trained model

Finally, we need to store the latent vectors and model reconstruction in order to compute the different metrics and plot the graphs that we use for ou analysis. A script that compute this can be run through the following command:
```
sbatch clinicadl_scripts/run_predict.sh $MAPS_DIRECTORY $CAPS_ROOT_DIRECTORY $TSV_DIRECTORY
```
with `$CAPS_ROOT_DIRECTORY` the root directory where all the simulated test sets are stored, and `$TSV_DIRECTORY` the folder that stored the tsv files with the splits.

## How to use

### Reconstruction

```
python src/reconstruction_figure.py ${MAPS_DIRECTORY} --split ${SPLIT}
```

### Comparison of simulation framework

```
python src/violin_plot.py ${MAPS_DIRECTORY} --split ${SPLIT}
```

### Healthiness of reconstructed images

```
python src/healthiness.py ${MAPS_DIRECTORY} --split ${SPLIT}
```

### Anomaly score

```
python src/anomaly_score.py ${MAPS_DIRECTORY} --split ${SPLIT}
```

### Latent Space Analysis

```
python src/latent_space.py
```

```
python src/closest_participant.py
```

### Linear mixed effect model
Some scripts needs to be run preliminary. To come.
```
python src/lmm.py
```

### Run all analysis

```
python src/main.py  ${MAPS_DIRECTORY} --split ${SPLIT}
```
