from natsort import natsorted
from pathlib import Path
import numpy as np
from tabulate import tabulate
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    evaluation_path = "/home/o340n/projects/2023-konstanz/data/2023.09.05-community1_vtk_independent_testdata/clean_by_species"
    evaluation_pattern = "classification_TE64_28C_CEL"

    npy_files = natsorted(
        list(Path(evaluation_path).rglob("*" + evaluation_pattern + "*"))
    )
    class_names = [

        "outliers",
        "13_Scenedesmus_obliquus",
        "17_Pediastrum_boryanum",
        "21-P_Chlorella_vulgaris",
        "23_Monoraphidium_obtusum",
        "40-aP_Monoraphidium_minutum",
        "CRP_Chlamydomonas_reinhardtii",
        "outliers",
    ]

    ## per file analysis
    for npy_file in npy_files:
        classification = np.genfromtxt(npy_file)
        if classification.size > 0:
            values = np.unique(classification)
            values_full = np.arange(-1, np.max(values) + 1)
            counts_full = np.zeros_like(values_full)
            for i, value in enumerate(values_full):
                counts_full[i] = np.sum(classification == value)
            counts_relative = counts_full / np.sum(counts_full)

            print(str(npy_file)[evaluation_path.__len__() :])
            table = [class_names, values_full, counts_full, counts_relative]
            print(tabulate(table))

    ## per folder analysis
    folders = [
        name
        for name in os.listdir(evaluation_path)
        if os.path.isdir(os.path.join(evaluation_path, name))
    ]
    folders = natsorted(folders)
    classification_combined = []
    target_combined = []
    for i_f, folder in enumerate(folders):
        print(str(i_f))
        npy_files = natsorted(
            list(
                Path(os.path.join(evaluation_path, folder)).rglob(
                    "*" + evaluation_pattern + "*"
                )
            )
        )
        classification = []
        for npy_file in npy_files:
            classification = classification + list(
                np.atleast_1d(np.genfromtxt(npy_file))
            )
        # print(classification.__len__())
        classification_combined = classification_combined + classification
        target_combined = target_combined + [i_f for _ in classification]
        values = np.unique(np.asarray(classification))
        values_full = np.arange(-1, np.max(values) + 1)
        counts_full = np.zeros_like(values_full)
        for i, value in enumerate(values_full):
            counts_full[i] = np.sum(classification == value)
        counts_relative = counts_full / np.sum(counts_full)

        print(folder)
        table = [class_names, counts_full, counts_relative]
        print(tabulate(table))

    print(
        classification_report(
            target_combined,
            classification_combined,
            # target_names=["outliers"] + folders,
            digits=2,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(
        target_combined, classification_combined, normalize="true"
    )
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={"float_kind": float_formatter})
    print(matrix * 100)
    matrix = confusion_matrix(target_combined, classification_combined)
    print(matrix)

    matrix = matrix[:-1, :-1]

    table = [
        class_names[1:],
        ["{:.2f}".format(i) for i in matrix.diagonal() / (matrix.sum(axis=1) + 1e-6)],
    ]
    print(tabulate(table))
