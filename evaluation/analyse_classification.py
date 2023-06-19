from natsort import natsorted
from pathlib import Path
import numpy as np
from tabulate import tabulate
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    evaluation_path = "/home/dlpc/data/anh/2023.05.23-16C-community-trainingsdata/trainingdata_split/test"
    evaluation_pattern = "_inliersmedian_normalize_cut_64_0_8_4_10_test_classification_multiSAD_anh_community_28c_inlier_rotate_split_230526_171557_output_threshold_1.csv"

    npy_files = natsorted(list(Path(evaluation_path).rglob("*" + evaluation_pattern)))

    x = 1

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

            print(str(npy_file)[evaluation_path.__len__():])
            table = [values_full, counts_full, counts_relative]
            print(tabulate(table))

    ## per folder analysis
    folders = [name for name in os.listdir(evaluation_path) if os.path.isdir(os.path.join(evaluation_path, name))]
    folders = natsorted(folders)
    classification_combined = []
    target_combined = []
    for i_f, folder in enumerate(folders):
        npy_files = natsorted(list(Path(os.path.join(evaluation_path, folder)).rglob("*" + evaluation_pattern)))
        classification = []
        for npy_file in npy_files:
            classification = classification + list(np.atleast_1d(np.genfromtxt(npy_file)))
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
        table = [values_full, counts_full, counts_relative]
        print(tabulate(table))

    print(classification_report(target_combined, classification_combined, target_names=["outliers"] + folders,
                                digits=2, zero_division=0))
    matrix = confusion_matrix(target_combined, classification_combined)
    table = [["outliers"] + folders, ["{:.2f}".format(i) for i in matrix.diagonal() / (matrix.sum(axis=1) + 1e-6)]]
    print(tabulate(table))

