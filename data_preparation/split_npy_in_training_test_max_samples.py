from pathlib import Path
import numpy as np
import random
import os
from tqdm import tqdm

if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_outliers_64/outliers"
    output_path = "/home/o340n/projects/2023-konstanz/data/2023.08.17-community1-temperature-experiment/trainings_data_subset_with_outliers_64/outliers_split"
    train_split = 0.8
    max_samples = 5000

    folders = [
        name
        for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]

    for folder in folders:
        files = Path(os.path.join(data_path, folder)).rglob("*.npy")
        data = []
        for file in tqdm(files):
            bla = np.load(file)
            data = data + [bla]

        data = np.concatenate(data)
        if data.shape[0] < max_samples:
            print(folder + " has only " + str(data.shape[0]) + " samples!")

        indices = random.sample(range(data.shape[0]), min(data.shape[0], max_samples))
        training_indices = random.sample(
            indices, int(np.ceil(len(indices) * train_split))
        )
        test_indices = [i for i in indices if i not in training_indices]

        training_data = data[training_indices, :, :, :]
        test_data = data[test_indices, :, :, :]

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
                output_path, "training", output_path_folder, folder + "_training.npy"
            ),
            training_data,
        )
        np.save(
            os.path.join(output_path, "test", output_path_folder, folder + "_test.npy"),
            test_data,
        )
