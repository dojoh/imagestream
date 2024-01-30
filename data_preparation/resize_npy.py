from pathlib import Path
import numpy as np
from misc.utils import *
from tqdm import tqdm
from natsort import natsorted


def cut_array(image, target_size):
    current_size = image.shape
    pad_rows = target_size[0] - current_size[2]
    pad_cols = target_size[1] - current_size[3]

    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    cropped_image = image[
        :,
        :,
        max(0, -pad_top) : current_size[2] - max(0, -pad_bottom),
        max(0, -pad_left) : current_size[3] - max(0, -pad_right),
    ]
    return cropped_image


if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.08.3-newdata/training_data/Species_Com2/sorted_2000/training"
    output_path = "/home/o340n/projects/2023-konstanz/data/2023.08.3-newdata/training_data/Species_Com2/sorted_2000/training_64"
    new_size = [64, 64]

    Path(output_path).mkdir(parents=True, exist_ok=True)
    files = list(Path(data_path).rglob("*.npy"))
    files = natsorted(files)

    for file in tqdm(files):
        array = np.load(file)
        new_array = cut_array(array, new_size)

        output_file = os.path.join(output_path, str(file)[len(data_path) + 1 :])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        np.save(output_file, new_array)
