import numpy as np
import os
from pathlib import Path
from utils import *
import glob
import pandas as pd


import random as rng



if __name__ == '__main__':
    cif_dir = '/home/o340n/projects/2023-konstanz/data/2023.08.3-newdata/training_data/Species_Com3/20C/Day_08/files/'

    channels = [0, 8, 4, 10]
    selection_type = [False, "outliers", "inliers"] #how to treat the selection?
    selection_type = selection_type[0] #0 = ignore, 1 = outliers, 2 = inliers
    # channels = [0, 8, 6, 7]
    selection_type = [False, "remainder", "selection"]
    # how to treat the selection.
    # False: ignore the selection
    # Remainder: write everything to npy except those in selection
    # Selection: only write those to npy that are in selection
    selection_type = selection_type[1]

    # selection_folder = "inliers"
    selection_folder = "outliers"

    vars = [
        0.01,
        0.026,
        0.026,
        0.026,
        0.02,
        0.026,
        0.026,
        0.026,
        0.005,
        0.026,
        0.05,
        0.026,
    ]
    # vars = [var*2 for var in vars]
    cut_size = 64
    final_size = cut_size
    max_number_per_cif = 50000

    clean_tasks = ["median", "normalize", "cut"]  # 'pca_orientation',

    name = "_".join(clean_tasks)
    outlier_name = selection_type if selection_type else ""
    export_name = (
        "_".join([outlier_name, selection_folder])
        + "_"
        + name
        + "_"
        + str(final_size)
        + "_"
        + "_".join(str(e) for e in channels)
    )

    data = list()
    labels = list()

    cif_files = Path(cif_dir).rglob("*.cif")

    init_javabridge()



    for cif_file in cif_files:
        class_data = list()

        selection = []
        for file in Path(os.path.join(cif_dir, selection_folder)).rglob("ids.csv"):
            df = pd.read_csv(file)
            selection.append(df)

        if selection:
            selection = pd.concat(selection, axis=0, ignore_index=True)
            selection = selection.drop_duplicates()
            selection = selection[
                selection.path == str(cif_file)[len(cif_dir) + 1 : -4]
            ]["Object Number"].tolist()

        if not selection_type:
            cif_data = load_cif(cif_file.as_posix(), channels=channels)
        elif selection_type == "remainder":
            cif_data = load_cif(
                cif_file.as_posix(), channels=channels, outliers=selection
            )
        else:
            cif_data = load_cif(
                cif_file.as_posix(), channels=channels, inliers=selection
            )

        if cif_data:
            # tmp_length = class_data.__len__()
            for clean_task in clean_tasks:
                if clean_task == "median":
                    cif_data = clean_median(cif_data, remove_hotpixels=True)
                elif clean_task == 'cut':
                    cif_data = clean_cut(cif_data, cut_size)
                elif clean_task == "cut_single":
                    cif_data = clean_cut_single(cif_data, cut_size)
                elif clean_task == "pca_orientation":
                    cif_data = clean_pca_orientation(cif_data)
                elif clean_task == "normalize":
                    cif_data = clean_normalize(cif_data, [vars[c] for c in channels])
                else:
                    assert False

            # fi(np.concatenate([cif_data[100][i, :, :] for i in range(4)], axis=1))

            # hf = h5py.File(os.path.join(cif_file.parent, cif_file.stem + '_' + export_name + '.h5')os.path.join(cif_file.parent, cif_file.stem + '_' + export_name + '.h5'), 'w')

            if max_number_per_cif > 0 and cif_data.__len__() > max_number_per_cif:
                indices = rng.sample(range(cif_data.__len__()), max_number_per_cif)
            else:
                indices = list(range(cif_data.__len__()))

            data = [cif_data[i].astype("half") for i in indices]

            np.save(
                os.path.join(
                    cif_file.parent, cif_file.stem + "_" + export_name + ".npy"
                ),
                np.stack(data),
            )
            # hf.create_dataset('data', data=data, maxshape=(None, data[0].shape[0], data[0].shape[1], data[0].shape[2]))

    javabridge.kill_vm()




