import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.get_backend()
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from natsort import natsorted
import collections
from pathlib import Path
import os
import shutil

if __name__ == '__main__':
    data_path = "/media/dojoh/data/projects/2023-konstanz/data/temperature_exp/com3/20C/Day_12"
    day = data_path[-2:]
    # day = data_path[-7:] #if /runX at the end
    # day = day.replace('/','_')
    runs_present = True
    # patterns = [['17_Pediastrum_boryanum_new_run5_Day_06', ['A01', 'B01', 'C01', 'D01',
    #                                                'A05', 'B05', 'C05', 'D05',
    #                                                'A09', 'B09', 'C09', 'D09']],
    #             ['17_Pediastrum_boryanum_new_run5_Day_08', ['A02', 'B02', 'C02', 'D02',
    #                                                'A06', 'B06', 'C06', 'D06',
    #                                                'A10', 'B10', 'C10', 'D10']],
    #             ['17_Pediastrum_boryanum_new_run5_Day_10', ['A03', 'B03', 'C03', 'D03',
    #                                                'A07', 'B07', 'C07', 'D07',
    #                                                'A11', 'B11', 'C11', 'D11']],
    #             ['17_Pediastrum_boryanum_new_run5_Day_12', ['A04', 'B04', 'C04', 'D04',
    #                                                'A08', 'B08', 'C08', 'D08',
    #                                                'A12', 'B12', 'C12', 'D12']],
    #             ]

    # Community 2
    # patterns = [['Sphaerocystis_sp_Day_' + day, ['A01', 'B01', 'C01', 'D01',
    #                                          'A04', 'B04', 'C04', 'D04',
    #                                          'A07', 'B07', 'C07', 'D07',
    #                                          'A10', 'B10', 'C10', 'D10']],
    #             ['C_augustae_Day_' + day, ['A02', 'B02', 'C02', 'D02',
    #                                    'A05', 'B05', 'C05', 'D05',
    #                                    'A08', 'B08', 'C08', 'D08',
    #                                    'A11', 'B11', 'C11', 'D11']],
    #             ['L_culleus_Day_' + day, ['A03', 'B03', 'C03', 'D03',
    #                                   'A06', 'B06', 'C06', 'D06',
    #                                   'A09', 'B09', 'C09', 'D09',
    #                                   'A12', 'B12', 'C12', 'D12']],
    #             ]
    patterns = [['C_minutissima_Day_' + day, ['A01', 'B01', 'C01', 'D01',
                                              'A05', 'B05', 'C05', 'D05']],
                ['C_luteoviridis_Day_' + day, ['A02', 'B02', 'C02', 'D02',
                                               'A06', 'B06', 'C06', 'D06']],
                ['S_intermedius_Day_' + day, ['A03', 'B03', 'C03', 'D03',
                                              'A07', 'B07', 'C07', 'D07']],
                ['S_quadricauda_Day_' + day, ['A04', 'B04', 'C04', 'D04',
                                              'A08', 'B08', 'C08', 'D08']],
                ]

    # sanity checks: are there any duplicates?
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
            row = ord(well[:1]) - 65
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

        # make all the folders
        # for well, name in pattern_dict.items():
        #     Path(os.path.join(data_path, name)).mkdir(parents=True, exist_ok=True)

        image_files = natsorted(list(Path(data_path).rglob("*.*")))
        image_files = [image_file for image_file in image_files if os.path.isfile(image_file)]

        for image_file in image_files:
            parts = image_file.name.split('.')
            if parts.__len__() == 2:
                if parts[0] in pattern_dict.keys():
                    print("moving " + image_file.name + " to " + pattern_dict[parts[0]])
                    # shutil.move(image_file, os.path.join(data_path, pattern_dict[parts[0]], image_file.name))
                    if runs_present:
                        os.rename(image_file, os.path.join(data_path,  # pattern_dict[parts[0]],
                                                           '_'.join([pattern_dict[parts[0]],
                                                                     str(image_file)[len(data_path):].split(os.sep)[0],
                                                                     image_file.name])))
                    else:
                        os.rename(image_file, os.path.join(data_path,  # pattern_dict[parts[0]],
                                                           '_'.join([pattern_dict[parts[0]], image_file.name])))


else:
    print('ciao!')
