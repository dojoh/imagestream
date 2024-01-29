import PIL
from PIL import ImageOps
import os
from pathlib import Path
import numpy as np
from collections import Counter
from natsort import natsorted
import difflib
import random
from operator import mul
from functools import reduce
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import time


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(
        np.pad(
            img,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    )


# import seaborn_image as isns
def write_images(images, filename, channels, grid_size=[1, 10], image_size=64):
    hues = [
        "white",
        "yellowgreen",
        "yellow",
        "orange",
        "red",
        "deeppink",
        "deepskyblue",
        "yellowgreen",
        "white",
        "orange",
        "red",
        "deeppink",
    ]
    out = PIL.Image.new(
        "RGB",
        (image_size * grid_size[1], image_size * grid_size[0] * channels.__len__()),
        color="black",
    )
    minclips = [-2, -2, 0, 0]
    maxclips = [2, 2, 2, 2]
    for i_image, image in enumerate(images):
        for idx, i_channel in enumerate(channels):
            minclip = minclips[idx]
            maxclip = maxclips[idx]
            row = i_image // grid_size[1]
            column = i_image % grid_size[1]
            tmp = PIL.Image.fromarray(
                (
                    (np.clip((image[idx, :, :] - minclip) / (maxclip - minclip), 0, 1))
                    * 255
                ).astype(np.uint8)
            )
            tmp = ImageOps.colorize(tmp, black="black", white=hues[i_channel])
            out.paste(tmp, (i_image * image.shape[2], idx * image.shape[1]))
            out.paste(
                tmp,
                (
                    column * image.shape[2],
                    idx * image.shape[1] + row * image.shape[1] * channels.__len__(),
                ),
            )
    parent = Path(filename).parent
    os.makedirs(parent, exist_ok=True)
    if images.__len__() > 0:
        out.save(filename)


class npyData:
    def __init__(self, folder_npy):
        self.files = list(Path(folder_npy).rglob("*.npy"))
        self.files = natsorted(self.files)
        self.file_lengths = []
        self.file_readers = []
        self.file_class = []
        class_folder_index = Path(folder_npy).parts.__len__()

        # init files
        for file in self.files:
            self.file_readers.append(np.load(file, mmap_mode="r"))
            self.file_lengths.append(self.file_readers[-1].shape[0])
            self.file_class.append(
                file.parts[class_folder_index]
            )  # the first folder inside the given data_path

        self.class_names = natsorted(list(set(self.file_class)))

        self.idx_label = []
        self.idx_file_reader = []
        self.idx_file_idx = []

        for i_file in range(self.files.__len__()):
            current_class = self.class_names.index(self.file_class[i_file])
            current_file_reader = self.file_readers[i_file]
            self.idx_label.extend([current_class] * self.file_lengths[i_file])
            self.idx_file_reader.extend(
                [current_file_reader] * self.file_lengths[i_file]
            )
            self.idx_file_idx.extend(list(range(self.file_lengths[i_file])))

        counter = Counter(self.idx_label)
        self.class_counts = np.array(
            [x for _, x in sorted(zip(counter.keys(), counter.values()))]
        )


if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.09.05-community1_vtk_independent_testdata/clean/"
    # data_path = "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/28C_split_2000/test"
    pattern = "multiSAD_anh_community_sep_c_plus_pediastrum_inlier_rotate_split_2000_rotate_230706_132556_output_distance_threshold_1"
    pattern = "classification_tsne"
    pattern = "classification_mbc_anh_community_16c_plus_pediastrum_inlier_rotate_split_2000_rotate_128_230726_104745_output_threshold_5"
    pattern = "classification_multiSAD_anh_community_16c_plus_pediastrum_inlier_rotate_split_2000_rotate_128_230726_104802_output_distance_threshold_0_3"
    pattern = "classification_TE64_allC_CEL_w_alloutliers_enet_v2s_aug_230929_143023"

    output_dir = "results_clean"
    name = ["cel", "outliers"]

    classes = [
        "13_Scenedesmus_obliquus",
        "17_Pediastrum_boryanum",
        "21-P_Chlorella_vulgaris",
        "23_Monoraphidium_obtusum",
        "40-aP_Monoraphidium_minutum",
        "CRP_Chlamydomonas_reinhardtii",
        "outliers",
    ]

    class_counts = []

    vis_channels = [0, 8, 4, 10]
    images_per_page = [3, 30]

    path = Path(data_path)
    output_path = os.path.join(path.parent, output_dir, pattern)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        Path(os.path.join(output_path, class_name)).mkdir(parents=True, exist_ok=True)

    for file in natsorted(Path(data_path).rglob("*" + pattern + "*")):
        print(file)
        class_counts_per_file = [0] * (classes.__len__() + 1)
        class_counts_per_file[0] = "_".join([file.parent.name, file.name])
        # class_counts_per_file[0] = file.name.split("_inliers")[0]
        npy_candidates = [x.name for x in Path(file.parent).rglob("*.npy")]
        npy_file = difflib.get_close_matches(file.name, npy_candidates, n=1, cutoff=0)

        image_data = np.load(os.path.join(file.parent, npy_file[0]))
        classification_data = np.atleast_1d(np.genfromtxt(file))
        unique, counts = np.unique(classification_data, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts_per_file[int(u) + 1] += c

        class_counts.append(class_counts_per_file)

        for i_class in range(classes.__len__()):
            ids = sorted(
                random.sample(
                    list(np.where(classification_data == i_class)[0]),
                    min(
                        reduce(mul, images_per_page, 1),
                        np.where(classification_data == i_class)[0].__len__(),
                    ),
                )
            )
            write_images(
                [image_data[i] for i in ids],
                os.path.join(output_path, classes[i_class], file.name + ".png"),
                vis_channels,
                grid_size=images_per_page,
            )

    print(tabulate([["name"] + classes, *class_counts]))

    df = pd.DataFrame(class_counts, columns=["name"] + classes)
    df.iloc[:, 1:] = df.iloc[:, 1:].div(df.sum(axis=1), axis=0)

    fig, axs = plt.subplots(figsize=(20, 12))

    format = ".2%"
    b = sns.heatmap(
        df.iloc[:, 1:].T,
        annot=True,
        fmt=format,
        ax=axs,
        xticklabels=df.iloc[:, 0],
        annot_kws={"rotation": 90, "fontsize": 6},
    )
    # b.set_xticklabels(b.get_xticklabels(), size=6)
    plt.tight_layout()
    plt.title(pattern)
    plt.show()

    plt.savefig(
        os.path.join(
            output_path,
            "confusion_matrix_" + name[0] + "_predicts_" + name[1] + ".png",
        ),
        bbox_inches="tight",
        dpi=150,
    )
