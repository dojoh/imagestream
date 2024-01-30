import os
from pathlib import Path
import pandas as pd
from natsort import natsorted
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    base_dir = '/home/dojoh/projects/2023-awi-plankton/data/231212_Ole_TiffFiles/extracted_features_16'
    feature_file = '/home/dojoh/projects/2023-awi-plankton/data/231212_Ole_TiffFiles/awi_240123_154714_features.txt'

    df = pd.read_csv(feature_file, header=None, )
    df = df.add_prefix('feature_')
    df = df.multiply(1000)

    files = natsorted(list(Path(base_dir).glob('*.npy')))

    data = [np.load(f, mmap_mode='r') for f in files]

    lengths = [d.shape[0] for d in data]
    feature_length = df.shape[0]

    assert np.sum(lengths) == feature_length, "somethings messed up with the feature file and the npy files"

    for i in tqdm(range(len(lengths))):
        tmp = df.iloc[:lengths[i], :]
        tmp.reset_index(inplace=True)
        df = df.iloc[lengths[i]:, :]
        assert tmp.shape[0] == lengths[i], "somethings wrong again..."
        tmp_filename = os.path.join(files[i].parent, files[i].stem + '.txt')
        tmp.to_csv(tmp_filename, index_label="Object Number", decimal='.', sep="\t")
