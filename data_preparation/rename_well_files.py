import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
import collections
from pathlib import Path
import os
import shutil

if __name__ == '__main__':
    data_path = "/home/dojoh/projects/postdoc/local_data/2023.04.20-20C-communitt-trainingsdata/Day_12/"

    patterns = [['13_Scenedesmus_obliquus_Day_12', ['A10', 'B10', 'C10', 'D10']],
                ['21-P_Chlorella_vulgaris_Day_12', ['E10', 'F10', 'G10', 'H10']],
                ['17_Pediastrum_boryanum_Day_12', ['A11', 'B11', 'C11', 'D11']],
                ['40-aP_Monoraphidium_minutum_Day_12', ['E11', 'F11', 'G11', 'H11']],
                ['23_Monoraphidium_obtusum_Day_12', ['A12', 'B12', 'C12', 'D12']],
                ['CRP_Chlamydomonas_reinhardtii_Day_12', ['E12', 'F12', 'G12', 'H12']]
                ]

    #sanity checks: are there any duplicates?
    all_wells = [p for pattern in patterns for p in pattern[1]]
    duplicate_wells = [item for item, count in collections.Counter(all_wells).items() if count > 1]
    if duplicate_wells:
        print("The following wells are present in two classes: ")
        print(', '.join(duplicate_wells))


    labels = []
    rows = []
    columns = []
    pattern_dict = {}

    for pattern in patterns:
        name = pattern[0]
        wells = pattern[1]
        for well in wells:
            pattern_dict[well] = name
            row = ord(well[:1])-65
            column = int(well[1:])
            labels.append(name)
            rows.append(row)
            columns.append(column)


    d = {'col': columns, 'row': rows, 'Name': labels}
    df = pd.DataFrame(data=d)
    colors = sns.color_palette("bright")
    colors.extend(sns.color_palette("deep"))
    sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(24, 14))
    g = sns.scatterplot(data=df, x="col", y="row", hue="Name", s=200)
    g.invert_yaxis()
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    plt.axis('equal')

    plt.show()

    x = input('If the mapping is correct, should we move the files? yes/no')

    if x == "yes":
        print('here we go...')

        #make all the folders
        for well, name in pattern_dict.items():
            Path(os.path.join(data_path, name)).mkdir(parents=True, exist_ok=True)

        image_files = natsorted(list(Path(data_path).rglob("*.*")))
        image_files = [image_file for image_file in image_files if os.path.isfile(image_file) ]

        for image_file in image_files:
            parts = image_file.name.split('.')
            if parts.__len__() == 2:
                if parts[0] in pattern_dict.keys():
                    print("moving " + image_file.name + " to " + pattern_dict[parts[0]])
                    # shutil.move(image_file, os.path.join(data_path, pattern_dict[parts[0]], image_file.name))
                    os.rename(image_file, os.path.join(data_path, pattern_dict[parts[0]], '_'.join([pattern_dict[parts[0]], image_file.name])))
    else:
        print('ciao!')


