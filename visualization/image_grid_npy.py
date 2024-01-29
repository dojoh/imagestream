from pathlib import Path
from tqdm import tqdm
import random
from misc.utils import *
import os


def ceildiv(a, b):
    return -(a // -b)


if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.09.05-community1_vtk_independent_testdata/data/28C/by_species"

    output_dir = "vis_"
    images = [5, 40]
    channels = [0, 8, 4, 10]

    if os.path.isdir(data_path):
        files = list(Path(data_path).rglob("*.npy"))
    else:
        files = [data_path]

    for file in tqdm(files):
        path, filename = os.path.split(file)
        outname = Path(file).stem

        f = np.load(file, mmap_mode="r")

        random.seed(10)
        print(f.shape[0])

        file_images = images.copy()
        file_images[0] = min(images[0], ceildiv(f.shape[0], images[1]))

        image_ids = random.sample(
            range(f.shape[0]), min(f.shape[0], file_images[0] * file_images[1])
        )
        image_ids = image_ids + [-1] * (np.prod(file_images) - len(image_ids))
        image_ids = np.reshape(image_ids, (file_images[0], file_images[1]))
        image_rows = []

        f = np.pad(f, ((0, 1), (0, 0), (0, 0), (0, 0)), constant_values=-2)

        for row in range(file_images[1]):
            image_rows.append(
                np.concatenate(
                    [
                        np.transpose(
                            separate_channels(
                                np.transpose(f[idx, :, :, :], (2, 1, 0)),
                                channels=channels,
                            ),
                            (1, 0, 2),
                        )
                        for idx in image_ids[:, row]
                    ]
                )
            )

        image = np.concatenate(image_rows, axis=1)

        # image = np.clip((image + 2)/4, 0, 1)
        im = Image.fromarray(np.uint8(image), "RGB")
        im.save(
            os.path.join(
                path, outname + "_" + "_".join([str(i) for i in images]) + ".png"
            )
        )
