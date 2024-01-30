import numpy as np
import os
from pathlib import Path
from utils import *
import glob
import pandas as pd

import random as rng

if __name__ == '__main__':
    cif_dir = '/home/dojoh/projects/2023-awi-plankton/data/231212_Ole_TiffFiles/'

    channels = [0, 8, 4, 10]
    channels = [2, 5, 6]
    selection_type = [False, "outliers", "inliers"]  # how to treat the selection?
    selection_type = selection_type[0]  # 0 = ignore, 1 = outliers, 2 = inliers
    # channels = [0, 8, 6, 7]

    vars = [4000, 4000, 400, 4000, 4000, 4000, 4000]
    # vars = [var*2 for var in vars]
    cut_size = 64
    final_size = 64
    max_number_per_file = 50000

    clean_tasks = ["median", "normalize", "cut"]

    name = '_'.join(clean_tasks)
    outlier_name = selection_type if selection_type else ""
    export_name = outlier_name + name + '_' + str(final_size) + '_' + '_'.join(str(e) for e in channels)

    data = list()
    labels = list()

    zip_files = Path(cif_dir).glob('*.zip')

    for zip_file in zip_files:
        class_data = list()

        selection = []
        for file in glob.glob(os.path.join(zip_file.parent, "selection_*_ids.csv")):
            df = pd.read_csv(file)
            selection.append(df)
        if selection:
            selection = pd.concat(selection, axis=0, ignore_index=True)
            selection = selection[selection.file == zip_file.stem]["Object Number"].tolist()

        if not selection_type:
            cif_data = load_zip(zip_file.as_posix(), channels=channels)
        elif selection_type == "outliers":
            cif_data = load_zip(zip_file.as_posix(), channels=channels, outliers=selection)
        else:
            cif_data = load_zip(zip_file.as_posix(), channels=channels, inliers=selection)

        # tmp_length = class_data.__len__()
        for clean_task in clean_tasks:
            if clean_task == 'median':
                cif_data = clean_median(cif_data, remove_hotpixels=False)
            elif clean_task == 'cut':
                cif_data = clean_cut(cif_data, cut_size)
            elif clean_task == 'cut_single':
                cif_data = clean_cut_single(cif_data, cut_size)
            elif clean_task == 'pca_orientation':
                cif_data = clean_pca_orientation(cif_data)
            elif clean_task == 'normalize':
                cif_data = clean_normalize(cif_data, [vars[c] for c in channels])
            else:
                assert (False)

        # fi(np.concatenate([cif_data[100][i, :, :] for i in range(4)], axis=1))

        # hf = h5py.File(os.path.join(cif_file.parent, cif_file.stem + '_' + export_name + '.h5')os.path.join(cif_file.parent, cif_file.stem + '_' + export_name + '.h5'), 'w')

        if max_number_per_file > 0 and cif_data.__len__() > max_number_per_file:
            indices = rng.sample(range(cif_data.__len__()), max_number_per_file)
        else:
            indices = list(range(cif_data.__len__()))

        data = [cif_data[i].astype('half') for i in indices]

        np.save(os.path.join(zip_file.parent, zip_file.stem + '_' + export_name + '.npy'),
                np.stack(data))

