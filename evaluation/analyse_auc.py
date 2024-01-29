import os
from pathlib import Path
import numpy as np
from natsort import natsorted
from tabulate import tabulate
from collections import Counter
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib

np.set_printoptions(precision=3)


def get_gt(folder, classes):
    npy_files = natsorted(list(Path(folder).rglob("*.npy")))

    # sanity check
    labels = []
    for npy_file in npy_files:
        tmp = np.load(npy_file, mmap_mode="r")
        c = [
            i
            for i, val in enumerate([c in npy_file.parent.name for c in classes])
            if val
        ]
        if c:
            c = c[0]
        else:
            c = -1
        labels += [c] * tmp.shape[0]
    labels = np.asarray(labels)
    return labels


def get_labels(files):
    preds = []
    for file in files:
        output = np.load(file[0])
        if file[1] == "multisad":
            output = -output  # other way around
        elif file[1] == "mbc":
            output = output  # other way around
        elif file[1] == "cel":
            output = output  # [:, :6]
        else:
            print("i dont know this..." + file[1])
        preds.append(np.squeeze(output))

    return preds


def make_binary(labels, type):
    new_labels = []
    for il, label in enumerate(labels):
        if type[il] == "multisad":
            cl = np.argmax(label, axis=1)
            maximum = np.max(label, axis=1)
            cl[maximum < -0.4] = label.shape[1]
        elif type[il] == "mbc":
            cl = np.argmax(label, axis=1)
            maximum = np.max(label, axis=1)
            cl[maximum < 3] = label.shape[1]
        elif type[il] == "cel":
            cl = np.argmax(label, axis=1)
            maximum = np.max(label, axis=1)
            # cl[maximum < 0] = label.shape[1]
        else:
            print("i dont know this..." + type[il])
        new_labels.append(cl)
    return new_labels


def compute_outlier_labels(labels):
    new_labels = []
    for il, label in enumerate(labels):
        new_labels.append(
            np.append(label, ((-np.max(label, axis=1)[:, np.newaxis])), axis=1)
        )
    return new_labels


if __name__ == "__main__":
    overwrite_files = [
        "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_CEL_w_alloutliers_enet_v2s_230823_091242_output.npy",
        "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_CEL_wo_alloutliers_enet_v2s_230823_091242_output.npy",
        "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_mBC_w_alloutliers_enet_v2s_230823_091242_output.npy",
        "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_mBC_wo_alloutliers_enet_v2s_230823_091245_output.npy",
        "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_mSAD_w_alloutliers_enet_v2s_230823_091245_output_distance.npy",
        "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_mSAD_wo_alloutliers_enet_v2s_230823_091242_output_distance.npy",
    ]
    # overwrite_files = [
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_20C_CEL_w_alloutliers_enet_v2s_230823_091209_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_20C_CEL_wo_alloutliers_enet_v2s_230823_091207_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_20C_mBC_w_alloutliers_enet_v2s_230823_091207_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_20C_mBC_wo_alloutliers_enet_v2s_230823_091208_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_20C_mSAD_w_alloutliers_enet_v2s_230823_091208_output_distance.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_20C_mSAD_wo_alloutliers_enet_v2s_230823_091207_output_distance.npy",
    # ]
    # overwrite_files = [
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_16C_CEL_w_alloutliers_enet_v2s_230823_091207_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_16C_CEL_wo_alloutliers_enet_v2s_230823_091208_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_16C_mBC_w_alloutliers_enet_v2s_230823_091222_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_16C_mBC_wo_alloutliers_enet_v2s_230823_091226_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_16C_mSAD_w_alloutliers_enet_v2s_230823_091228_output_distance.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_16C_mSAD_wo_alloutliers_enet_v2s_230823_091247_output_distance.npy",
    # ]
    # overwrite_files = [
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_28C_CEL_w_alloutliers_enet_v2s_230823_091253_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_28C_CEL_wo_alloutliers_enet_v2s_230823_091228_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_28C_mBC_w_alloutliers_enet_v2s_230823_091228_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_28C_mBC_wo_alloutliers_enet_v2s_230823_091228_output.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_28C_mSAD_w_alloutliers_enet_v2s_230823_091239_output_distance.npy",
    #     "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_28C_mSAD_w_alloutliers_enet_v2s_230823_091239_output_distance.npy",
    # ]
    extension = "_allC"
    result_files = [
        [
            [],
            "cel",
            "CEL_w_outliers" + extension,
        ],
        [
            [],
            "cel",
            "CEL_wo_outliers" + extension,
        ],
        [
            [],
            "mbc",
            "mBC_w_outliers" + extension,
        ],
        [
            [],
            "mbc",
            "mBC_wo_outliers" + extension,
        ],
        [
            [],
            "multisad",
            "multiSAD_w_outliers" + extension,
        ],
        [
            [],
            "multisad",
            "multiSAD_wo_outliers" + extension,
        ],
    ]
    for i in range(len(result_files)):
        result_files[i][0] = overwrite_files[i]
    data_folder_test = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/test"

    gt_classes = [
        "13_Scenedesmus_obliquus",
        "17_Pediastrum_boryanum",
        "21-P_Chlorella_vulgaris",
        "23_Monoraphidium_obtusum",
        "40-aP_Monoraphidium_minutum",
        "CRP_Chlamydomonas_reinhardtii",
        "outliers",
    ]

    # gt_classes = [
    #     "C_augustae",
    #     "L_culleus",
    #     "Sphaerocystis_sp",
    #     "outliers_com2",
    #     "outliers_com2",
    #     "outliers_com2",
    #     "outliers_com2",
    # ]
    # get gt
    label_gt = get_gt(data_folder_test, gt_classes)
    label_preds = get_labels(result_files)
    label_preds[0] = label_preds[0][:, :6]  # ignore new class for now
    label_binary = make_binary(label_preds, [i[1] for i in result_files])
    label_preds = compute_outlier_labels(label_preds)

    for label_pred in label_preds:
        assert len(label_gt) == label_pred.shape[0], "missmatch!"

    classes = list(range(len(gt_classes)))

    values, counts = np.unique(label_gt, return_counts=True)
    table = [gt_classes, counts]
    print(tabulate(table))

    font = {"family": "serif", "size": 22}

    matplotlib.rc("font", **font)

    for cls in classes:
        if cls < 6:
            # I don't allways want to see the curves for all classes. just outliers
            continue
        print(str(cls))
        fig, ax = plt.subplots()
        for i_l, label_pred in enumerate(label_preds):
            print(i_l)
            class_gt = label_gt == cls

            fpr, tpr, thresholds = roc_curve(class_gt, label_pred[:, cls], pos_label=1)
            score = -fpr + tpr
            threshold_id = np.argmax(score)
            optimal_threshold = thresholds[threshold_id]

            print(
                "optimal threshold for "
                + result_files[i_l][2]
                + " is:"
                + str(optimal_threshold)
            )

            PrecisionRecallDisplay.from_predictions(
                class_gt,
                label_pred[:, cls],
                name=" ".join([gt_classes[cls], result_files[i_l][2]]),
                ax=ax,
            )
        _ = ax.set_title("PrecisionRecallDisplay " + gt_classes[cls])

    for cls in classes:
        if cls < 6:
            continue
        fig, ax = plt.subplots()
        for i_l, label_pred in enumerate(label_preds):
            class_gt = label_gt == cls
            RocCurveDisplay.from_predictions(
                class_gt,
                label_pred[:, cls],
                name=" ".join([gt_classes[cls], result_files[i_l][2]]),
                ax=ax,
            )
        _ = ax.set_title("RocCurveDisplay " + gt_classes[cls])
    for i_l, label_pred in enumerate(label_preds):
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            label_gt, label_binary[i_l], ax=ax, normalize="true"
        )
        _ = ax.set_title("Confusion Matrix " + result_files[i_l][2])
    plt.show()
