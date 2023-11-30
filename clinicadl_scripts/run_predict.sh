#!/bin/bash
#SBATCH --job-name=predict_test
#SBATCH --output=slurm_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00


MAPS_DIR=$1
CAPS_ROOT=$2
TSV_DIR=$3
echo $MAPS_DIR
# Predict on test AD
GROUPE=test_AD
CAPS_DIR=${CAPS_ROOT}/caps_pet_uniform/
PARTICIPANT_TSV=${TSV_DIR}/test/AD_baseline.tsv

echo clinicadl predict $MAPS_DIR $GROUPE --caps_directory $CAPS_DIR --participants_tsv $PARTICIPANT_TSV --diagnoses AD --split 0 --selection_metrics loss --save_latent_tensor  --save_tensor  --overwrite
clinicadl   predict $MAPS_DIR $GROUPE \
            --caps_directory $CAPS_DIR \
            --participants_tsv $PARTICIPANT_TSV \
            --diagnoses AD \
            -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 \
            --selection_metrics loss \
            --save_latent_tensor  \
            --save_tensor \
            --overwrite

# Predict on test CN
GROUPE=test_CN
PARTICIPANT_TSV=${TSV_DIR}/test/CN-test_baseline.tsv

echo clinicadl predict $GROUPE
clinicadl   predict $MAPS_DIR $GROUPE \
            --caps_directory $CAPS_DIR \
            --participants_tsv $PARTICIPANT_TSV \
            --diagnoses CN \
            -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 \
            --selection_metrics loss \
            --save_latent_tensor  \
            --save_tensor \
            --overwrite

# Predict on all hypometabolic CAPS
PATHOLOGY_LIST=(bvftd lvppa svppa)
for PATHOLOGY in ${PATHOLOGY_LIST[@]} ; do
    GROUPE=test_hypo_${PATHOLOGY}_30
    CAPS_DIR=${CAPS_ROOT}/hypometabolic_caps/caps_${PATHOLOGY}_30

    echo clinicadl predict $GROUPE
    clinicadl   predict $MAPS_DIR $GROUPE \
                --caps_directory $CAPS_DIR \
                --participants_tsv $PARTICIPANT_TSV \
                --diagnoses CN \
                -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 \
                --selection_metrics loss \
                --save_latent_tensor  \
                --save_tensor \
                --overwrite
    echo making tsv
done

PERCENTAGE_LIST=(5 10 15 20 30 40 50 70)
for PERCENTAGE in ${PERCENTAGE_LIST[@]} ; do
    GROUPE=test_hypo_ad_${PERCENTAGE}
    CAPS_DIR=${CAPS_ROOT}/hypometabolic_caps/caps_ad_${PERCENTAGE}

    echo clinicadl predict $GROUPE
    clinicadl   predict $MAPS_DIR $GROUPE \
                --caps_directory $CAPS_DIR \
                --participants_tsv $PARTICIPANT_TSV \
                --diagnoses CN \
                -s 0 -s 1 -s 2 -s 3 -s 4 -s 5 \
                --selection_metrics loss \
                --save_latent_tensor  \
                --save_tensor \
                --overwrite
    echo making tsv
    python healthiness_measurment.py $MAPS_DIR ad $PERCENTAGE
done