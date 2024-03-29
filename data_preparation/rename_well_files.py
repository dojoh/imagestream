import pandas as pd
import seaborn as sns
import matplotlib

# matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

from natsort import natsorted
import collections
from pathlib import Path
import os
import shutil

if __name__ == '__main__':
    data_path = "/home/o340n/projects/2023-konstanz/data/2023.08.3-newdata/training_data/Species_Com2/raw/20C/Day_06/"
    runs_present = True
    # Community 2
    patterns = [['Sphaerocystis_sp_Day_06', ['A01', 'B01', 'C01', 'D01',
                                             'A04', 'B04', 'C04', 'D04',
                                             'A07', 'B07', 'C07', 'D07',
                                             'A10', 'B10', 'C10', 'D10']],
                ['C_augustae_Day_06', ['A02', 'B02', 'C02', 'D02',
                                       'A05', 'B05', 'C05', 'D05',
                                       'A08', 'B08', 'C08', 'D08',
                                       'A11', 'B11', 'C11', 'D11']],
                ['L_culleus_Day_06', ['A03', 'B03', 'C03', 'D03',
                                      'A06', 'B06', 'C06', 'D06',
                                      'A09', 'B09', 'C09', 'D09',
                                      'A12', 'B12', 'C12', 'D12']],
                ]
    # patterns = [['C_minutissima_Day_12', ['A01', 'B01', 'C01', 'D01',
    #                                       'A05', 'B05', 'C05', 'D05']],
    #             ['C_luteoviridis_Day_12', ['A02', 'B02', 'C02', 'D02',
    #                                        'A06', 'B06', 'C06', 'D06']],
    #             ['S_intermedius_Day_12', ['A03', 'B03', 'C03', 'D03',
    #                                       'A07', 'B07', 'C07', 'D07']],
    #             ['S_quadricauda_Day_12', ['A04', 'B04', 'C04', 'D04',
    #                                       'A08', 'B08', 'C08', 'D08']],
    #             ]

    # sanity checks: are there any duplicates?
    all_wells = [p for pattern in patterns for p in pattern[1]]
    duplicate_wells = [
        item for item, count in collections.Counter(all_wells).items() if count > 1
    ]
    if duplicate_wells:
        print("The following wells are present in two classes: ")
        print(", ".join(duplicate_wells))

    labels = []
    rows = []
    columns = []
    pattern_dict = {}

    for pattern in patterns:
        name = pattern[0]
        wells = pattern[1]
        for well in wells:
            pattern_dict[well] = name
            row = ord(well[:1]) - 65
            column = int(well[1:])
            labels.append(name)
            rows.append(row)
            columns.append(column)

    d = {"col": columns, "row": rows, "Name": labels}
    df = pd.DataFrame(data=d)
    colors = sns.color_palette("bright")
    colors.extend(sns.color_palette("deep"))
    sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(24, 14))
    g = sns.scatterplot(data=df, x="col", y="row", hue="Name", s=200)
    g.invert_yaxis()
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ["A", "B", "C", "D", "E", "F", "G", "H"])
    plt.axis("equal")
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.001)

    # plt.show(block=False)

    # x = input("If the mapping is correct, should we move the files? yes/no")
    x = "yes"
    if x == "yes":
        print("here we go...")

        # make all the folders
        # for well, name in pattern_dict.items():
        #     Path(os.path.join(data_path, name)).mkdir(parents=True, exist_ok=True)

        image_files = natsorted(list(Path(data_path).rglob("*.*")))
        image_files = [
            image_file for image_file in image_files if os.path.isfile(image_file)
        ]

        for image_file in image_files:
            parts = image_file.name.split(".")
            if parts.__len__() == 2:
                if parts[0] in pattern_dict.keys():
                    print("moving " + image_file.name + " to " + pattern_dict[parts[0]])
                    # shutil.move(image_file, os.path.join(data_path, pattern_dict[parts[0]], image_file.name))
                    if runs_present:
                        os.rename(image_file, os.path.join(data_path, #pattern_dict[parts[0]],
                                                           '_'.join([pattern_dict[parts[0]], str(image_file)[len(data_path):].split(os.sep)[0], image_file.name])))
                    else:
                        os.rename(image_file, os.path.join(data_path, #pattern_dict[parts[0]],
                                                           '_'.join([pattern_dict[parts[0]], image_file.name])))


else:
    print("ciao!")
