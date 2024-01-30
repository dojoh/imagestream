from pathlib import Path
from tqdm import tqdm
import random
from misc.utils import *
import os
from torchvision import transforms
import torch


class MultiChannelSharpness:
    # the beautiful sharpnesstransform of pytorch assumes the iamge to be between 0 and 1. niceone.
    def __init__(self, sharpness_factor):
        self.sharpnesstransform = transforms.RandomAdjustSharpness(
            sharpness_factor=sharpness_factor
        )

    def __call__(self, img):
        transformed_channels = []

        for idx, channel in enumerate(img):
            # print(idx)
            # lets assume the image is allways between -10 and 10. doesnt really matter that much.
            channel = self.sharpnesstransform((channel[None, :, :] + 10) / 20)
            channel = channel * 20 - 10
            transformed_channels.append(channel)

        img = torch.cat(transformed_channels)

        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


def ceildiv(a, b):
    return -(a // -b)


def augment(i, transform, clamp):
    image = transform(torch.tensor(i).double())

    torch.clamp(image, clamp[0], clamp[1])

    image = image.numpy()

    return image


if __name__ == "__main__":
    config = {}

    config["rotate"] = None
    config["reflect_h"] = None
    config["reflect_v"] = None
    config["sharpness"] = None
    config["noise_sigma"] = None
    config["scale_sigma"] = None
    config["add_sigma"] = None
    config["clamp_min"] = -2
    config["clamp_max"] = 2

    config["rotate"] = True
    config["reflect_h"] = True
    config["reflect_v"] = True
    config["sharpness"] = 7
    config["noise_sigma"] = 0.02
    config["scale_sigma"] = 0.2
    config["add_sigma"] = 0.01
    config["clamp_min"] = -2
    config["clamp_max"] = 2

    data_path = "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/combined_2000/training/17_Pediastrum_boryanum/17_Pediastrum_boryanum_training_16C.npy"

    # output_dir = "vis_augmentations_"
    images = [20, 40]
    channels = [0, 8, 4, 10]

    if os.path.isdir(data_path):
        files = list(Path(data_path).rglob("*.npy"))
    else:
        files = [data_path]

    transform = []
    if config["reflect_h"]:
        transform.append(transforms.RandomHorizontalFlip())

    if config["reflect_v"]:
        transform.append(transforms.RandomVerticalFlip())

    if config["rotate"]:
        transform.append(
            transforms.RandomRotation(
                degrees=180, interpolation=transforms.InterpolationMode.BILINEAR
            )
        )
    if config["sharpness"]:
        # trans
        transform.append(MultiChannelSharpness(sharpness_factor=config["sharpness"]))

    if config["noise_sigma"]:
        transform.append(
            transforms.Lambda(
                lambda image: image
                + np.random.normal(0, config["noise_sigma"], image.shape).astype(
                    np.float32
                )
            )
        )
    if config["scale_sigma"]:
        transform.append(
            transforms.Lambda(
                lambda image: image
                * np.random.normal(
                    1 - config["scale_sigma"] / 2,
                    config["scale_sigma"],
                    [image.shape[0], 1, 1],
                ).astype(np.float32)
            )
        )

    if config["add_sigma"]:
        transform.append(
            transforms.Lambda(
                lambda image: image
                + np.random.normal(
                    0, config["add_sigma"], [image.shape[0], 1, 1]
                ).astype(np.float32)
            )
        )

    transform = transforms.Compose(transform)

    for file in tqdm(files):
        path, filename = os.path.split(file)
        outname = Path(file).stem

        f = np.load(file, mmap_mode="r")

        random.seed(10)
        print(f.shape[0])
        images[0] = min(images[0], ceildiv(f.shape[0], images[1]))

        image_ids = random.sample(range(f.shape[0]), min(f.shape[0], images[0]))
        image_ids = np.repeat(image_ids, images[1]).tolist()
        image_ids = image_ids + [-1] * (np.prod(images) - len(image_ids))
        image_ids = np.reshape(image_ids, (images[0], images[1]))
        image_rows = []

        f = np.pad(f, ((0, 1), (0, 0), (0, 0), (0, 0)), constant_values=-2)

        for row in range(images[1]):
            image_rows.append(
                np.concatenate(
                    [
                        np.transpose(
                            separate_channels(
                                np.transpose(
                                    augment(
                                        f[idx, :, :, :],
                                        transform,
                                        [config["clamp_min"], config["clamp_max"]],
                                    ),
                                    (2, 1, 0),
                                ),
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
                path,
                outname
                + "_"
                + "_".join([str(i) for i in images])
                + "_augmentations.png",
            )
        )
