from natsort import natsorted
from pathlib import Path
import numpy as np
from tabulate import tabulate
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    matplotlib.get_backend()

    evaluation_paths = [["16c",
                         "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/16C_split_2000/test"],
                        ["20c",
                         "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/20C_split_2000/test"],
                        ["28c",
                         "/home/o340n/projects/2023-konstanz/data/2023.06.06-temperature_trainingsdata_plus_pediastrum/28C_split_2000/test"]]
    evaluation_patterns = ["16c", "20c", "28c", "all_c"]

    folders = [name for name in os.listdir(evaluation_paths[0][1]) if
               os.path.isdir(os.path.join(evaluation_paths[0][1], name))]
    folders = natsorted(folders)
    classes = ["outliers"] + folders

    evaluation_df = None
    for (evaluation_path_name, evaluation_path) in evaluation_paths:
        for evaluation_pattern in evaluation_patterns:
            npy_files = natsorted(list(Path(evaluation_path).rglob("*" + evaluation_pattern + "*.csv")))

            x = 1

            ## per file analysis
            # for npy_file in npy_files:
            #     classification = np.genfromtxt(npy_file)
            #     if classification.size > 0:
            #         values = np.unique(classification)
            #         values_full = np.arange(-1, classes.__len__() - 1)
            #         counts_full = np.zeros_like(values_full)
            #         for i, value in enumerate(values_full):
            #             counts_full[i] = np.sum(classification == value)
            #         counts_relative = counts_full / np.sum(counts_full)
            #
            #         print(str(npy_file)[evaluation_path.__len__():])
            #         table = [classes, counts_full, counts_relative]
            #         print(tabulate(table))

            ## per folder analysis
            folders = [name for name in os.listdir(evaluation_path) if
                       os.path.isdir(os.path.join(evaluation_path, name))]
            folders = natsorted(folders)
            classification_combined = []
            target_combined = []
            for i_f, folder in enumerate(folders):
                npy_files = natsorted(
                    list(Path(os.path.join(evaluation_path, folder)).rglob("*" + evaluation_pattern + "*")))
                classification = []
                for npy_file in npy_files:
                    classification = classification + list(np.atleast_1d(np.genfromtxt(npy_file)))
                # print(classification.__len__())
                classification_combined = classification_combined + classification
                target_combined = target_combined + [i_f for _ in classification]
                values = np.unique(np.asarray(classification))
                values_full = np.arange(-1, np.max(values) + 1)
                counts_full = np.zeros_like(values_full)
                for i, value in enumerate(values_full):
                    counts_full[i] = np.sum(classification == value)
                counts_relative = counts_full / np.sum(counts_full) * 100

                print(folder)
                table = [classes, counts_full, counts_relative]
                print(tabulate(table))

            print(
                classification_report(target_combined, classification_combined,
                                      labels=[-1] + list(range(folders.__len__())),
                                      target_names=["outliers"] + folders,
                                      digits=2, zero_division=0))
            report = classification_report(target_combined, classification_combined,
                                           labels=[-1] + list(range(folders.__len__())),
                                           target_names=["outliers"] + folders,
                                           digits=2, zero_division=0, output_dict=True)
            # report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"],
            #                             "support": report['macro avg']['support']}})
            report_df = pd.DataFrame(report).transpose()
            report_df['training_data'] = evaluation_pattern
            report_df['test_data'] = evaluation_path_name
            if (report_df.index == 'micro avg').any():
                report_df = report_df.drop('micro avg')
            if (report_df.index == 'accuracy').any():
                report_df = report_df.drop('accuracy')

            report_df = report_df.drop('weighted avg')
            report_df = report_df.drop('macro avg')
            report_df = report_df.drop('outliers')
            report_df['class'] = report_df.index

            if isinstance(evaluation_df, pd.DataFrame):
                evaluation_df = pd.concat([evaluation_df, report_df])
            else:
                evaluation_df = report_df
    # evaluate everything :)

    # df = df.drop(df[df].index)
    for value in ['f1-score', 'precision', 'recall', 'support']:
        fig, axs = plt.subplots(ncols=evaluation_df['class'].unique().__len__() + 1,
                                gridspec_kw=dict(width_ratios=[3] * evaluation_df['class'].unique().__len__() + [0.2]))
        for id, class_name in enumerate(evaluation_df['class'].unique()):
            print(class_name)
            tmp = evaluation_df.loc[evaluation_df['class'] == class_name].pivot(index='test_data',
                                                                                columns='training_data',
                                                                                values=value)
            print(tmp)
            max_length = 15
            class_name = info = (class_name[:max_length] + '..') if len(class_name) > max_length else class_name
            vmin = 0
            vmax = 1
            format = '.2f'
            if value == 'support':
                vmax = 3000
                format = 'g'

            if id == 0:
                sns.heatmap(tmp, annot=True, cbar=False, ax=axs[id], vmin=vmin, vmax=vmax, fmt=format)
                axs[id].set_xlabel('')
            else:
                sns.heatmap(tmp, annot=True, yticklabels=False, cbar=False, ax=axs[id], vmin=vmin, vmax=vmax,
                            fmt=format)
                axs[id].set_ylabel('')
                axs[id].set_xlabel('')
            axs[id].set_title(class_name, rotation=90)
        fig.colorbar(axs[0].collections[0], cax=axs[-1])
        fig.suptitle(value, fontsize=16)
        fig.text(0.5, 0.04, 'training_data', ha='center')
    plt.tight_layout()
    plt.show()
