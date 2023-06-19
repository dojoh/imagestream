import os
from pathlib import Path
import numpy as np
from natsort import natsorted
from tabulate import tabulate
from collections import Counter
import torch

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
            self.file_readers.append(np.load(file, mmap_mode='r'))
            self.file_lengths.append(self.file_readers[-1].shape[0])
            self.file_class.append(file.parts[class_folder_index])  # the first folder inside the given data_path

        self.class_names = natsorted(list(set(self.file_class)))

        self.idx_label = []
        self.idx_file_reader = []
        self.idx_file_idx = []

        for i_file in range(self.files.__len__()):
            current_class = self.class_names.index(self.file_class[i_file])
            current_file_reader = self.file_readers[i_file]
            self.idx_label.extend([current_class] * self.file_lengths[i_file])
            self.idx_file_reader.extend([current_file_reader] * self.file_lengths[i_file])
            self.idx_file_idx.extend(list(range(self.file_lengths[i_file])))

        counter = Counter(self.idx_label)
        self.class_counts = np.array([x for _, x in sorted(zip(counter.keys(), counter.values()))])

if __name__ == "__main__":
    folder_npy_train = "/home/dojoh/projects/postdoc/local_data/2023.04.20-20C-community-trainingsdata/data/"
    file_sad = "/home/dojoh/projects/postdoc/local_data/2023.04.20-20C-community-trainingsdata/results/multiSAD_anh_community_20c_230425_112916_output.npy"
    ckpt_model = "/home/dojoh/Desktop/transfer/deeplearning_limno/saved/multiSAD_anh_community_20c/230425_112916/model/model_best.pth"

    if folder_npy_train:
        npy_train = npyData(folder_npy_train)
    threshold = 1

    samples = 100
    images_per_page = 10
    vis_channels = [0, 8, 4, 10]
    name = Path(file_sad).stem

    # os.makedirs(Path(file_sad).parent, exist_ok=True)
    output_file = os.path.join(Path(file_sad).parent, name + "_classification_threshold_" + str(threshold) + '.csv')


    ckpt = torch.load(ckpt_model, map_location=torch.device('cpu'))
    target = ckpt['models']['model']['target'].cpu().numpy()

    sad_data = np.load(file_sad)
    sad_squared_distance = np.sum((sad_data - target[None, :, None]) ** 2, axis=1)

    sad_decision_threshold = sad_squared_distance < threshold

    sad_decision = np.argmin(sad_squared_distance, axis=1)
    sad_decision[np.sum(sad_decision_threshold, axis=1) == 0] = -1

    counter = Counter(sad_decision)
    class_counts = np.array([x for _, x in sorted(zip(counter.keys(), counter.values()))])

    if folder_npy_train:
        print(tabulate([["class names", "outliers"] + npy_train.class_names,
                        ["prediction"] + [i for i in class_counts],
                        ["ground truth", 0] + [i for i in npy_train.class_counts]], tablefmt='orgtbl'))
    else:
        print(tabulate([["class names", "outliers"] + [i for i in range(class_counts.__len__()-1)],
                        ["prediction"] + [i for i in class_counts]], tablefmt='orgtbl'))

    np.savetxt(output_file, sad_decision.astype(int), fmt='%i', delimiter=",")
