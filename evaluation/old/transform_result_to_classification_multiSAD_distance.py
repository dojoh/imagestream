import os
from pathlib import Path
import numpy as np
from natsort import natsorted
from tabulate import tabulate
from collections import Counter
import torch
from tqdm import tqdm
#28C done
#
if __name__ == "__main__":
    file_sad = "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/results/28C_multiSAD_anh_community_sep_c_plus_pediastrum_inlier_rotate_split_2000_rotate_230706_132556_output_distance.npy"
    data_folder_test = "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/28C"

    threshold = 1

    # samples = 100
    # images_per_page = 10
    vis_channels = [0, 8, 4, 10]
    name = Path(file_sad).stem

    # os.makedirs(Path(file_sad).parent, exist_ok=True)
    # output_file = os.path.join(Path(file_sad).parent, name + "_classification_threshold_" + str(threshold) + '.csv')

    sad_squared_distance = np.load(file_sad)

    sad_decision_threshold = sad_squared_distance < threshold

    sad_decision = np.argmin(sad_squared_distance, axis=1)
    sad_decision[np.sum(sad_decision_threshold, axis=1) == 0] = -1

    counter = Counter(sad_decision)
    class_counts = np.array([x for _, x in sorted(zip(counter.keys(), counter.values()))])

    print(tabulate([["class names", "outliers"] + [i for i in range(class_counts.__len__() - 1)],
                    ["prediction"] + [i for i in class_counts]], tablefmt='orgtbl'))

    npy_files = natsorted(list(Path(data_folder_test).rglob("*.npy")))

    #sanity check
    number_of_images = 0
    for npy_file in npy_files:
        tmp = np.load(npy_file, mmap_mode='r')
        number_of_images += tmp.shape[0]

    if number_of_images == sad_decision.__len__():
        print('all seems to fit!')
    else:
        assert False, 'the number of files from the result file does not match that of that of the npy files in the folder?!'

    sad_decision_write_list = sad_decision
    for npy_file in tqdm(npy_files):
        tmp = np.load(npy_file, mmap_mode='r')
        file_length = tmp.shape[0]
        results = sad_decision_write_list[:file_length]
        sad_decision_write_list = sad_decision_write_list[file_length:]
        output_file = os.path.join(npy_file.parent, str(npy_file.stem) + "_classification_" + name + "_threshold_" + str(threshold) + ".csv" )

        np.savetxt(output_file, results.astype(int), fmt='%i', delimiter=",")
