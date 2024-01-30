import os
import shutil
import numpy as np
import scipy as sp

# import torch
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from math import log2
import javabridge
import javabridge.jutil
import bioformats
import bioformats.formatreader

# import skvideo.io

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import (
    ColorBar,
    LinearColorMapper,
    WheelZoomTool,
    Label,
    ColumnDataSource,
    Slider,
    CustomJS,
    RangeSlider,
)
from PIL import Image
from PIL import ImageOps
import PIL


def ensure_dir(dir_name: str):
    """Creates folder if not exists."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def entropy(events, ets=1e-15):
    maximum = np.array(np.ones(events.shape)) / events.__len__()
    maximum = -sum([p * log2(p + ets) for p in maximum])
    return max(0, np.array(-sum([p * log2(p + ets) for p in events])) / maximum)


def create_certainty_matrix(y_true, y_pred, train_classnames, classnames):
    # Build confusion matrix
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    filter = y_true < train_classnames.__len__()

    known_true = y_true[filter]
    known_pred = y_pred[:, filter]

    # handle known
    correct = known_pred == known_true
    ratio = np.sum(correct, axis=0)

    ratio = np.append(ratio, [0 for i in range(y_pred.shape[0] + 1)])
    known_true = np.append(known_true, [i for i in range(y_pred.shape[0] + 1)])

    cf_matrix = confusion_matrix(known_true, ratio, normalize="true")
    cf_matrix = cf_matrix[0 : train_classnames.__len__(), 0 : y_pred.shape[0] + 1]
    cf_matrix = np.pad(
        cf_matrix,
        ((0, 0), (0, max(0, y_pred.shape[0] + 1 - cf_matrix.shape[1]))),
        "constant",
        constant_values=(0),
    )

    # handle unknown
    for i in range(classnames.__len__()):
        current_class = i
        filter = y_true == current_class
        unknown_pred = y_pred[:, filter]
        counts = sp.stats.mode(unknown_pred, axis=0).count
        certainty = [np.count_nonzero(counts == y) for y in range(y_pred.shape[0] + 1)]
        if np.sum(certainty) > 0:
            certainty = certainty / np.sum(certainty)
        cf_matrix = np.vstack([cf_matrix, certainty])

    ent = np.zeros((cf_matrix[:, 0].shape[0], 1))
    for i in range(cf_matrix.shape[0]):
        ent[i] = entropy(cf_matrix[i, :])

    cf_matrix = np.hstack([cf_matrix, ent])
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
    #                      columns=[i for i in classes])
    df_cm = pd.DataFrame(
        cf_matrix * 100,
        index=[i for i in train_classnames] + ["cert" + str(i) for i in classnames],
        columns=[
            str(i) + "/" + str(y_pred.shape[0]) for i in range(y_pred.shape[0] + 1)
        ]
        + ["entropy"],
    )
    # print(df_cm)
    plt.figure(figsize=(24, 14))
    ax = sn.heatmap(
        df_cm, cmap="crest", annot=True, vmin=0, vmax=100
    )  # vmax=np.sum(cf_matrix)/classes.__len__())

    ax.set(xlabel="predicted", ylabel="actual")
    return ax.get_figure()


def create_confusion_matrix(y_true, y_pred, classes, normalization="true"):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize=normalization)

    additional_classnames = cf_matrix.shape[0] - classes.__len__()
    for i in range(additional_classnames):
        classes.extend(str(i))

    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
    #                      columns=[i for i in classes])
    if normalization is None:
        df_cm = pd.DataFrame(
            cf_matrix, index=[i for i in classes], columns=[i for i in classes]
        )
        plt.figure(figsize=(24, 14))
        ax = sn.heatmap(
            df_cm, annot=True, vmin=0, vmax=max(cf_matrix.flatten())
        )  # vmax=np.sum(cf_matrix)/classes.__len__())
    else:
        df_cm = pd.DataFrame(
            cf_matrix * 100, index=[i for i in classes], columns=[i for i in classes]
        )
        # print(df_cm)
        plt.figure(figsize=(24, 14))
        ax = sn.heatmap(
            df_cm, cmap="crest", annot=True, vmin=0, vmax=100
        )  # vmax=np.sum(cf_matrix)/classes.__len__())
    ax.set(xlabel="predicted", ylabel="actual")
    return ax.get_figure()


def create_scatter_plot(
    x_data,
    y_data,
    hue=None,
    logx=False,
    logy=False,
    x_label=False,
    y_label=False,
    classifier=False,
):
    # Build confusion matrix
    plt.figure(figsize=(24, 14))

    x_data_np = np.asarray(x_data).astype(float).transpose()
    y_data_np = np.asarray(y_data).astype(float).transpose()

    # fig, ax = plt.subplots()

    ax = sn.scatterplot(x=x_data_np, y=y_data_np, hue=hue, s=2)
    if logx:
        ax.set(xscale="log")
    if logy:
        ax.set(yscale="log")

    if classifier:
        DecisionBoundaryDisplay.from_estimator(
            classifier,
            np.array([x_data_np, y_data_np]).transpose(),
            ax=ax.axes,
            response_method="predict",
            plot_method="contour",
            grid_resolution=1000,
            # xlabel=iris.feature_names[0],
            # ylabel=iris.feature_names[1],
            # shading="auto",
        )

    if x_label:
        ax.set(xlabel=x_label)
    if y_label:
        ax.set(ylabel=y_label)
    else:
        ax.set(xlabel="predicted", ylabel="actual")
    return ax.get_figure()


def create_joint_plot(
    x_data,
    y_data,
    hue=None,
    logx=False,
    logy=False,
    x_label="x",
    y_label="y",
    classifier=False,
):
    # Build confusion matrix
    sn.set(rc={"figure.figsize": (24, 14)})

    x_data_np = np.asarray(x_data).astype(float).transpose()
    y_data_np = np.asarray(y_data).astype(float).transpose()

    # if logx:
    #     x_data_np = np.log10(x_data_np)
    # if logy:
    #     y_data_np = np.log10(y_data_np)

    data = {x_label: x_data_np, y_label: y_data_np, "class": hue}
    pddata = pd.DataFrame(data)

    jp = sn.jointplot(
        data=pddata,
        x=x_label,
        y=y_label,
        hue="class",
        s=1
        # kind='kde',
        # fill=True,
    )

    log_scale = logx and logy

    jp.plot_joint(sn.kdeplot, hue="class", fill=True, alpha=0.5, log_scale=log_scale)
    jp.plot_marginals(sn.histplot, bins=50, log_scale=log_scale)

    jp.figure.set_size_inches(24, 14)

    return jp.figure.get_figure()  # get_figure()


def create_ridge_plot(y_true, y_pred, normalization="true"):
    # Build confusion matrix

    sn.set_theme(style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0)})
    df = pd.DataFrame(
        dict(
            x=np.asarray(y_pred).astype(float).flatten(),
            g=np.asarray(y_true).astype(int).transpose(),
        )
    )

    # Initialize the FacetGrid object
    classes = np.unique(y_true)
    pal = sn.cubehelix_palette(classes.__len__(), rot=-0.25, light=0.7)
    g = sn.FacetGrid(df, row="g", hue="g", aspect=15, height=1, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sn.kdeplot, "x", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5
    )
    g.map(sn.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    # g.tick_params(axis='y')

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.65)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(ylabel="")
    g.despine(left=True, bottom=True)

    g.set(xlabel="predicted")
    return g.figure.get_figure()


def fi(image, image_range=False):
    p = figure(
        tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
        sizing_mode="stretch_both",
        title="hi",
        match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,crosshair",
        toolbar_location="right",
    )
    p.sizing_mode = "scale_both"
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    color_mapper = LinearColorMapper(
        palette="Greys256", low=min(image.flatten()), high=max(image.flatten())
    )

    if image.shape.__len__() < 3:
        p.image(
            image=[image],
            x=0,
            y=0,
            dw=image.shape[1],
            dh=image.shape[0],
            color_mapper=color_mapper,
        )
    elif image.shape[2] == 3:
        img = np.empty((image.shape[0], image.shape[1]), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((image.shape[0], image.shape[1], 4))

        if not image_range:
            output = (
                (image - min(image.flatten()))
                / (max(image.flatten()) - min(image.flatten()))
                * 255
            )
        else:
            output = (image - image_range[0]) / (image_range[1] - image_range[0]) * 255

        view[:, :, 0] = output[:, :, 0]
        view[:, :, 1] = output[:, :, 1]
        view[:, :, 2] = output[:, :, 2]
        view[:, :, 3] = 255 * np.ones_like(view[:, :, 0])

        p.image_rgba(
            image=[img], x=[0], y=[0], dw=[image.shape[1]], dh=[image.shape[0]]
        )
    elif image.shape[2] == 12:
        output = np.zeros((image.shape[0] * 2, image.shape[1] * 6))
        for i in range(image.shape[2]):
            column, row = divmod(i, 6)
            # row = 1 - row
            output[
                column * image.shape[0] : ((column + 1) * image.shape[0]),
                row * image.shape[1] : ((row + 1) * image.shape[1]),
            ] = image[:, :, i]

            my_text = Label(
                x=row * image.shape[1] + 1,
                y=column * image.shape[0] + 1,
                text="Ch.: " + str(i),
                text_color="red",
                text_font_size="1em",
            )
            p.add_layout(my_text)
        p.image(
            image=[output],
            x=0,
            y=0,
            dw=output.shape[1],
            dh=output.shape[0],
            color_mapper=color_mapper,
        )

    else:
        p.image(
            image=[image],
            x=0,
            y=0,
            dw=image.shape[1],
            dh=image.shape[0],
            color_mapper=color_mapper,
        )

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)

    p.add_layout(color_bar, "right")
    show(p)


def fi_slide(sequence):
    N = sequence.__len__()

    p = figure(
        tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
        plot_width=1900,
        plot_height=1000,
        title="hi",
        match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,crosshair",
        toolbar_location="below",
    )

    sequence_bokeh = []

    clip_values = [-1, 1]

    for i in range(N):
        img = np.empty((sequence[i].shape[0:2]), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape(
            (sequence[i].shape[0], sequence[i].shape[1], 4)
        )
        for x in range(sequence[i].shape[0]):
            for y in range(sequence[i].shape[1]):
                if sequence[i].dtype != np.uint8:
                    view[x, y, 0] = int(
                        (sequence[i][x, y, 0] - clip_values[0])
                        / (clip_values[1] - clip_values[0])
                        * 255
                    )
                    view[x, y, 1] = int(
                        (sequence[i][x, y, 1] - clip_values[0])
                        / (clip_values[1] - clip_values[0])
                        * 255
                    )
                    view[x, y, 2] = int(
                        (sequence[i][x, y, 2] - clip_values[0])
                        / (clip_values[1] - clip_values[0])
                        * 255
                    )
                else:
                    view[x, y, 0] = sequence[i][x, y, 0]
                    view[x, y, 1] = sequence[i][x, y, 1]
                    view[x, y, 2] = sequence[i][x, y, 2]

                view[x, y, 3] = 255
        sequence_bokeh.append(img)

    source = ColumnDataSource(data=dict(image=[sequence_bokeh[0]]))
    p.image_rgba(
        image="image",
        x=0,
        y=0,
        dw=sequence_bokeh[0].shape[1],
        dh=sequence_bokeh[0].shape[0],
        source=source,
    )
    # p.image_rgba(image=[sequence_bokeh[0]], x=0, y=0, dw=[sequence_bokeh[0].shape[1]], dh=[sequence_bokeh[0].shape[0]])

    slider = Slider(start=0, end=(N - 1), value=0, step=1, title="Frame")
    callback = CustomJS(
        args=dict(source=source, slider=slider, sequence=sequence_bokeh),
        code="""
        const data = source.data;
        const S = slider.value;
        source.data['image'] = [sequence[S]];
        source.change.emit();
    """,
    )
    slider.js_on_change("value", callback)

    show(column(p, slider))


def separate_channels(images, channels=[0, 8, 4, 10], type="npy"):
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

    output = [
        np.zeros((images.shape[0], images.shape[1], 3), dtype=np.uint8)
        for _ in range(images.shape[2])
    ]

    if type == "cif":
        minclip = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + 0.001
        maxclip = (
            np.array([0.1, 0.03, 0.1, 0.1, 0.03, 0.1, 0.1, 0.1, 0.1, 0.1, 0.03, 0.1])
            / 4
        )
    else:
        maxclip = (
            np.asarray([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5]) * 1.5
        )
        minclip = np.asarray([-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]) * 1.5
    # minclip = [-1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -1, -0.1, -0.1, -0.1]
    for i in range(images.shape[2]):
        i_channel = channels[i]
        tmp = PIL.Image.fromarray(
            (
                (
                    np.clip(
                        (images[:, :, i] - minclip[i_channel])
                        / (maxclip[i_channel] - minclip[i_channel]),
                        0,
                        1,
                    )
                )
                * 255
            ).astype(np.uint8)
        )
        tmp = ImageOps.colorize(tmp, black="black", white=hues[i_channel])
        output[i] = np.array(tmp.getdata(), dtype=np.uint8).reshape(
            tmp.size[1], tmp.size[0], 3
        )

    output = np.hstack(output)

    return output


def write_animation(path, sequence):
    video = np.stack(sequence, 0)
    skvideo.io.vwrite(path, video)


def init_javabridge():
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="8G")
    rootLoggerName = javabridge.get_static_field(
        "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;"
    )
    rootLogger = javabridge.static_call(
        "org/slf4j/LoggerFactory",
        "getLogger",
        "(Ljava/lang/String;)Lorg/slf4j/Logger;",
        rootLoggerName,
    )
    logLevel = javabridge.get_static_field(
        "ch/qos/logback/classic/Level", "ERROR", "Lch/qos/logback/classic/Level;"
    )
    javabridge.call(
        rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel
    )


def resize_with_padding(image, target_size, pad_color=0):
    current_size = image.shape
    pad_rows = target_size[0] - current_size[0]
    pad_cols = target_size[1] - current_size[1]

    pad_top = pad_rows // 2
    pad_bottom = pad_rows - pad_top
    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    cropped_image = image[
        max(0, -pad_top) : current_size[0] - max(0, -pad_bottom),
        max(0, -pad_left) : current_size[1] - max(0, -pad_right),
        :,
    ]

    if isinstance(pad_color, list):
        output = np.zeros(target_size + [image.shape[2]])
        assert image.shape[2] == len(
            pad_color
        ), "wrong number of per-channel pad values"

        for ic in range(image.shape[2]):
            output[..., ic] = np.pad(
                cropped_image[..., ic],
                [
                    (max(0, pad_top), max(0, pad_bottom)),
                    (max(0, pad_left), max(0, pad_right)),
                ],
                mode="constant",
                constant_values=pad_color[ic],
            )
        return output
    else:
        padded_image = np.pad(
            cropped_image,
            [
                (max(0, pad_top), max(0, pad_bottom)),
                (max(0, pad_left), max(0, pad_right)),
                (0, 0),
            ],
            mode="constant",
            constant_values=pad_color,
        )

    return padded_image
