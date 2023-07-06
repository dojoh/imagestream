import os
from pathlib import Path
import numpy as np
from natsort import natsorted
from tabulate import tabulate
from collections import Counter
import torch
from tqdm import tqdm
from glob import glob
import pandas as pd

#
if __name__ == "__main__":
    tsne_files = "/home/o340n/projects/2023-konstanz/data/2023.04.14-20C-community/results/2023_07_05_tsne_classification"
    data_folder_test = "/home/o340n/projects/2023-konstanz/data/2023.04.14-20C-community/data"
    image_file_suffix = "_median_normalize_cut_64_0_8_4_10.npy"

    vis_channels = [0, 8, 4, 10]
    name = Path(tsne_files).stem

    classes = glob(tsne_files + os.path.sep + '*' + os.path.sep, recursive=True)
    classes = [Path(x).name for x in classes]
    classes = natsorted(classes)

    data = None
    for i_cl, cl in enumerate(classes):
        ids = pd.read_csv(os.path.join(tsne_files, cl, "ids.csv"))
        ids['class'] = cl
        ids['class_id'] = i_cl
        if isinstance(data, pd.DataFrame):
            data = pd.concat([data, ids], ignore_index=True, sort=False)
        else:
            data = ids

    counter = Counter(data["class_id"])
    class_counts = np.array([x for _, x in sorted(zip(counter.keys(), counter.values()))])
    print(tabulate([["class names"] + [classes[i] for i in range(class_counts.__len__())],
                    ["prediction"] + [i for i in class_counts]], tablefmt='orgtbl'))

    image_files = glob(data_folder_test + os.path.sep + '**' + os.path.sep + '*' + image_file_suffix, recursive=True)

    for image_file in image_files:
        parts = image_file.split(os.path.sep)
        file_name = parts[-1][:-image_file_suffix.__len__()]
        folder_name = parts[-2]

        image_file_data = np.load(image_file, mmap_mode='r')
        elements_in_file = image_file_data.shape[0]
        output = [-1] * elements_in_file
        data_selection = data[(data["file"] == file_name) & (data['folder'] == folder_name)]

        for index, row in data_selection.iterrows():
            output[row["Object Number"]] = row["class_id"]
        output_name = image_file[:-4] + "_classification_tsne.csv"

        np.savetxt(output_name, np.asarray(output).astype(int), fmt='%i', delimiter=",")

