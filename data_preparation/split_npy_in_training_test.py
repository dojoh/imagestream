from pathlib import Path
import numpy as np
import random
import os
from tqdm import tqdm

if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_outliers_64/outliers"
    output_path = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_outliers_64/outliers_split"
    train_split = 0.8

    files = Path(data_path).rglob("*.npy")

    for file in tqdm(files):
        bla = np.load(file)

        training_indices = random.sample(
            range(bla.shape[0]), int(np.ceil(bla.shape[0] * train_split))
        )
        test_indices = [i for i in range(bla.shape[0]) if i not in training_indices]

        training_data = bla[training_indices, :, :, :]
        test_data = bla[test_indices, :, :, :]

        output_path_folder = str(file.parent)[data_path.__len__() :]
        if output_path_folder[0] == os.sep:
            output_path_folder = output_path_folder[1:]
        Path(os.path.join(output_path, "training", output_path_folder)).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(output_path, "test", output_path_folder)).mkdir(
            parents=True, exist_ok=True
        )

        np.save(
            os.path.join(
                output_path, "training", output_path_folder, file.stem + "_training.npy"
            ),
            training_data,
        )
        np.save(
            os.path.join(
                output_path, "test", output_path_folder, file.stem + "_test.npy"
            ),
            test_data,
        )
