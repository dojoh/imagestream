from pathlib import Path
import numpy as np
import os
from natsort import natsorted
import pandas as pd
from tabulate import tabulate


def pprint_df(dframe):
    print(tabulate(dframe, headers="keys", tablefmt="psql", showindex=False))


if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.09.05-community1_vtk_independent_testdata/data/28C/by_species"

    files = Path(os.path.join(data_path)).rglob("*.npy")
    files = natsorted(files)

    rows = []
    for file in files:
        # print(str(file))
        mmapped_array = np.load(file, mmap_mode="r")
        rows.append([str(file), file.name, mmapped_array.shape[0]])

    rows_df = pd.DataFrame(rows, columns=["folder", "file", "count"])
    pprint_df(rows_df[["file", "count"]])
    folders = [x[0] for x in os.walk(data_path)]

    for folder in folders:
        tmp_df = rows_df[rows_df["folder"].str.startswith(folder)]
        print(folder[len(data_path) :] + ": " + str(tmp_df.sum()["count"]))
