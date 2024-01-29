from pathlib import Path
from tqdm import tqdm
import random
from misc.utils import *
import bioformats
import bioformats.formatreader


def kill_javabridge():
    javabridge.kill_vm()


if __name__ == "__main__":
    data_path = "/home/o340n/projects/2023-sideeffect/data/2023.10.18-initial-data/raw/231018_Ole_A/Nitzschia_frigida_MAG20x_2.cif"

    output_dir = "vis_"
    images = [20, 10]
    channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    image_size = [64, 64]

    path, file = os.path.split(data_path)
    outname = Path(file).stem

    init_javabridge()
    with bioformats.ImageReader(data_path, perform_init=True) as reader:
        image_count = reader.rdr.getSeriesCount() // 2

        random.seed(10)
        print(image_count)
        images[0] = min(images[0], image_count // images[1])
        image_ids = random.sample(range(image_count), images[0] * images[1])

        image_ids = np.reshape(image_ids, (images[0], images[1]))
        image_rows = []

        for row in tqdm(range(images[1])):
            row_data = []
            for idx in image_ids[:, row]:
                tmp_data = []
                for channel in channels:
                    tmp_data.append(reader.read(series=idx * 2, c=channel))

                tmp_data = np.stack(tmp_data, axis=-1)
                colors = [
                    np.median(tmp_data[..., c]) for c in range(tmp_data.shape[-1])
                ]
                bla = resize_with_padding(
                    tmp_data,
                    image_size,
                    colors,
                )
                tmp_data = separate_channels(bla, channels=channels, type="cif")
                row_data.append(tmp_data)
            image_rows.append(np.concatenate(row_data))

        image = np.concatenate(image_rows, axis=1)

        # image = np.clip((image + 2)/4, 0, 1)
        im = Image.fromarray(np.uint8(image), "RGB")
        im.save(
            os.path.join(
                path, outname + "_" + "_".join([str(i) for i in images]) + ".png"
            )
        )
    kill_javabridge()
