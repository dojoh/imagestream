import seaborn as sn
import pandas as pd

import itertools
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib.path as mpltPath

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.palettes import d3, Spectral, Turbo256
from bokeh.transform import linear_cmap
import numpy as np
import time
from bokeh.models import HoverTool

alpha = 1 # 0.3



def get_colors(number_of_classes):
    if number_of_classes <= 10:
        colors = d3['Category10'][max(3, number_of_classes)]
    elif number_of_classes <= 20:
        colors = d3['Category20'][max(3, number_of_classes)]
    else:
        idx = np.round(np.linspace(0, 255, number_of_classes)).astype(int)
        colors = tuple([Turbo256[i] for i in idx])
    return colors

def create_scatter_plot(data, x=None, y=None, hue = None, logx = False, logy = False, x_label = False, y_label = False,
                        classifier=False, palette=None,
                        hue_order=None
                        ):
    # Build confusion matrix
    plt.figure(figsize=(24, 14))

    if not x:
        x = data.columns[0]
    if not y:
        y = data.columns[1]

    if palette:
        ax = sn.scatterplot(data=data,
                            x=x,
                            y=y,
                            hue=hue,
                            hue_order=hue_order,
                            s=5,
                            palette=palette)
    else:
        ax = sn.scatterplot(data=data,
                            x=x,
                            y=y,
                            hue=hue,
                            hue_order=hue_order,
                            s=5)
    if logx:
        ax.set(xscale="log")
    if logy:
        ax.set(yscale="log")

    if x_label:
        ax.set(xlabel=x)
    if y_label:
        ax.set(ylabel=y)
    else:
        ax.set(xlabel=x, ylabel=y)

    return ax.get_figure()


def create_ridge_plot(data, x=None, classes=None, row_order=None):
    # Build confusion matrix

    sn.set_theme(style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0)})

    if not x:
        x = data.columns[1]
    if not classes:
        classes = data.columns[0]

    # Initialize the FacetGrid object

    pal = sn.cubehelix_palette(classes.__len__(), rot=-.25, light=.7)
    g = sn.FacetGrid(data, row=classes, hue=classes, aspect=24, height=1.5, palette=pal, hue_order=row_order, row_order=row_order, col_order=row_order)

    # Draw the densities in a few steps
    g.map(sn.kdeplot, x,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sn.kdeplot, x, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
    # g.tick_params(axis='y')

    # Define and use a simple function to label the plot in axes coordinates
    def label(bla, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.65)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(ylabel="")
    g.despine(left=True, bottom=True)

    g.set(xlabel=x)
    return g.figure.get_figure()


def bokeh_scatter(x, y, classes, class_names=None, title=None):
    if not class_names:
        class_names = sorted(np.unique(classes))
    else:
        class_names = [str(c) for c in class_names]

    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "cat": [class_names[int(c)] for c in classes]
        }
    )

    # create figure and plot
    p = figure(title=title, sizing_mode='stretch_both')

    colors = get_colors(len(class_names))

    for class_value, color in zip(class_names, colors):
        source_filter = ColumnDataSource(df.loc[df['cat']==class_value])
        p.scatter(x='x', y='y',
                  color=color, legend_group='cat', source=source_filter,
                  alpha=alpha, line_width=0, size=10, muted_alpha=alpha*0.1)

    p.legend.click_policy = "mute"
    show(p)

def bokeh_scatter_continuous(x, y, z, classes, class_names=None, title=None):
    if not class_names:
        class_names = sorted(np.unique(classes))
    else:
        class_names = [str(c) for c in class_names]

    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "cat": [class_names[int(c)] for c in classes],
            "magnitude": z
        }
    )

    # create figure and plot
    p = figure(title=title, sizing_mode='stretch_both',
               tooltips=[
                   ("index", "$index"),
                   ("cat", "@cat"),
                   ("mag", "@magnitude"),
               ])

    colors = get_colors(len(class_names))
    markers = ['hex', 'circle', 'triangle', 'diamond', 'plus', 'inverted_triangle', 'square', 'star']

    cmap = linear_cmap(field_name='magnitude', palette="Turbo256", low=np.quantile(z, 0.02), high=np.quantile(z, 0.98))

    for class_value, marker in zip(class_names, itertools.cycle(markers)):
        source_filter = ColumnDataSource(df.loc[df['cat'] == class_value])
        p.scatter(x='x', y='y',
                  color=cmap, marker=marker,
                  legend_group='cat', source=source_filter,
                  alpha=alpha, line_width=0, size=10, muted_alpha=alpha*0.1)

    color_bar = ColorBar(color_mapper=cmap['transform'], width=10)
    p.add_layout(color_bar, 'right')

    p.legend.click_policy = "mute"
    show(p)



def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def select_points():
    pts = np.empty((0, 2))
    poly = None
    poly_corners = None
    while True:
        # tellme('Select polygon corners with left click, finish with right click')
        new_point = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=MouseButton.RIGHT))
        if new_point.size>0:
            pts = np.append(pts, new_point, axis=0)
            # print(pts)
            if poly:
                poly[0].remove()
                poly_corners[0].remove()
            poly = plt.fill(pts[:, 0], pts[:, 1], lw=2, alpha=0.3, color="#154c79")
            poly_corners = plt.plot(pts[:, 0], pts[:, 1], 'x', markersize=5, color="#154c79")

            time.sleep(0.01)  # Wait a second
            plt.draw()
        else:
            break
    # if poly:
    #     poly[0].remove()
    #     poly_corners[0].remove()
    return pts
def interactive_scatter(x, y, classes, class_names=None, title=None):
    if not class_names:
        class_names = sorted(np.unique(classes))
    else:
        class_names = [str(c) for c in class_names]

    tellme(title)

    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "Kategorie": [class_names[int(c)] for c in classes]
        }
    )

    colors = get_colors(len(class_names))

    plt.figure(2, figsize=(24, 14))

    sn.set_style("darkgrid")

    ax = sn.scatterplot(data=df,
                        x='x',
                        y='y',
                        hue='Kategorie',
                        palette=colors,
                        hue_order=class_names,
                        alpha=alpha,
                        linewidth=0,
                        s=25)
    # plt.grid()
    # axs = []
    # for kat in df["Kategorie"].unique():
    #     axs.append(sn.scatterplot(data=df.loc[df.Kategorie == kat],
    #                         x='x',
    #                         y='y',
    #                         hue='Kategorie',
    #                         # hue_order=hue_order,
    #                         s=5))

    polypoints = select_points()
    path = mpltPath.Path(polypoints)
    inside = path.contains_points(np.concatenate([np.reshape(x,(-1,1)),np.reshape(y,(-1,1))], axis=1))
    # inside = np.reshape(x > 0,(-1,1))
    # ax = plt.scatter(df[inside].x,df[inside].y, color='r')

    # sn.scatterplot(data=df[inside],
    #                x='x',
    #                y='y',
    #                hue='Kategorie',
    #                palette=colors,
    #                hue_order=class_names,
    #                s=35,
    #                legend=False)
    # time.sleep(0.01)
    plt.show(block=False)
    plt.draw()
    time.sleep(0.01)
    return inside, ax.get_figure()



