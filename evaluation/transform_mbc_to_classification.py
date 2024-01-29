import os
from pathlib import Path
import numpy as np
from natsort import natsorted
from tabulate import tabulate
from collections import Counter
import torch
from tqdm import tqdm

# 28C done
#
if __name__ == "__main__":
    output_file = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/results_test/TE64_allC_mBC_w_alloutliers_enet_v2s_230823_091242_output.npy"
    data_folder_test = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_alloutliers_64/allC/test"

    threshold = 2  # determined by ROC curve if outliers are included in training

    name = Path(output_file).stem

    output = np.load(output_file)

    classification = np.argmax(output, axis=1)
    classification[np.max(output, axis=1) < threshold] = -1  # outliers

    classes = [-1] + list(range(output.shape[1]))
    class_counts = [np.sum(classification == i) for i in classes]

    print(
        tabulate(
            [
                ["class names", "outliers"]
                + [i for i in range(class_counts.__len__() - 1)],
                ["prediction"] + [i for i in class_counts],
            ],
            tablefmt="orgtbl",
        )
    )

    npy_files = natsorted(list(Path(data_folder_test).rglob("*.npy")))

    # sanity check
    number_of_images = 0
    for npy_file in npy_files:
        tmp = np.load(npy_file, mmap_mode="r")
        number_of_images += tmp.shape[0]

    if number_of_images == classification.__len__():
        print("all seems to fit!")
    else:
        assert (
            False
        ), "the number of files from the result file does not match that of that of the npy files in the folder?!"

    decision_write_list = classification
    for npy_file in tqdm(npy_files):
        tmp = np.load(npy_file, mmap_mode="r")
        file_length = tmp.shape[0]
        results = decision_write_list[:file_length]
        decision_write_list = decision_write_list[file_length:]
        output_file = os.path.join(
            npy_file.parent,
            str(npy_file.stem)
            + "_classification_"
            + name
            + "_threshold_"
            + str(threshold).replace(".", "_")
            + ".csv",
        )

        np.savetxt(output_file, results.astype(int), fmt="%i", delimiter=",")
