from pathlib import Path
from natsort import natsorted
import numpy as np
import scipy.signal
import scipy.ndimage
from tqdm import tqdm
import inspect, sys
import os
from sklearn.decomposition import PCA
from scipy import ndimage
import math
import zipfile
import torch
from torchvision.transforms.functional import pad, resize
import torchvision.transforms as T
# import bioformats
# import bioformats.formatreader
# import javabridge
# import javabridge.jutil
import io
from bokeh.plotting import figure, show
from bokeh.models import ColorBar, LinearColorMapper, WheelZoomTool, Label
from tifffile import TiffFile
# import napari_segment_blobs_and_things_with_membranes as nsbatwm  # version 0.3.4


def fi(image, image_range=False):
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], match_aspect=True, width=1900, height=1000)
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    color_mapper = LinearColorMapper(palette="Greys256", low=min(image.flatten()), high=max(image.flatten()))

    if image.shape.__len__() < 3:
        p.image(image=[image], x=0, y=0, dw=image.shape[1], dh=image.shape[0], color_mapper=color_mapper)
    elif image.shape[2] == 3:
        img = np.empty((image.shape[0], image.shape[1]), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((image.shape[0], image.shape[1], 4))

        if not image_range:
            output = (image - min(image.flatten()))/(max(image.flatten())-min(image.flatten()))*255
        else:
            output = (image - image_range[0])/(image_range[1]-image_range[0])*255

        view[:, :, 0] = output[:, :, 0]
        view[:, :, 1] = output[:, :, 1]
        view[:, :, 2] = output[:, :, 2]
        view[:, :, 3] = 255*np.ones_like(view[:, :, 0])

        p.image_rgba(image=[img], x=[0], y=[0], dw=[image.shape[1]], dh=[image.shape[0]])
    elif image.shape[2] == 12:
        output = np.zeros((image.shape[0]*2, image.shape[1]*6))
        for i in range(image.shape[2]):
            column, row = divmod(i, 6)
            # row = 1 - row
            output[column*image.shape[0]:((column+1)*image.shape[0]), row*image.shape[1]:((row+1)*image.shape[1])] = image[:, :, i]

            my_text = Label(x=row*image.shape[1] + 1, y=column*image.shape[0] + 1, text = "Ch.: " + str(i),
                            text_color = "red", text_font_size =  "1em")
            p.add_layout(my_text)
        p.image(image=[output], x=0, y=0, dw=output.shape[1], dh=output.shape[0], color_mapper=color_mapper)

    else:
        p.image(image=[image], x=0, y=0, dw=image.shape[1], dh=image.shape[0], color_mapper=color_mapper)

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    p.add_layout(color_bar, 'right')

    show(p)

def cut_pad_image(image, center, max_size, bg = None):
    #in case it needs to be cut
    left =  max_size // 2 - center[1]
    right = center[1] + max_size // 2 - image.shape[2]
    top = max_size // 2 - center[0]
    bottom = center[0] + max_size // 2 - image.shape[1]
    if bg == None:
        image = pad(torch.FloatTensor(image), (left, top, right, bottom), 0, 'edge')
    else:
        image = pad(torch.FloatTensor(image), (left, top, right, bottom), bg, 'constant')

    # transform = T.Resize(size=(final_size, final_size), antialias=True)

    image = image.numpy()
    return image


def cut_pad_resize_image(image, center, cut_size, final_size):
    left = cut_size // 2 - center[1]
    right = center[1] + cut_size // 2 - image.shape[2]
    top = cut_size // 2 - center[0]
    bottom = center[0] + cut_size // 2 - image.shape[1]

    image = pad(torch.FloatTensor(image), (left, top, right, bottom), 0, 'edge')

    transform = T.Resize(size=(final_size, final_size), antialias=True)
    image = transform(image)

    image = image.numpy()

    return image


# def init_javabridge():
#     javabridge.start_vm(class_path=bioformats.JARS, max_heap_size='8G')
#     rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
#     rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
#                                         "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
#     logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "ERROR", "Lch/qos/logback/classic/Level;")
#     javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)
#
# def kill_javabridge():
#     javabridge.kill_vm()

def clean_cut(data, cut_size):
    for i_data in tqdm(range(data.__len__()), desc=sys._getframe().f_code.co_name + ": "):
        mask = np.amax(data[i_data], 2)

        # fi(mask)
        project_0 = np.sum(mask > np.quantile(mask.flatten(), 0.8), 1)
        project_1 = np.sum(mask > np.quantile(mask.flatten(), 0.8), 0)

        center = [0, 0]
        center[0] = np.nanmean([i for i in range(mask.shape[0]) if project_0[i] > 0])
        center[1] = np.nanmean([i for i in range(mask.shape[1]) if project_1[i] > 0])

        if np.isnan(center[0]):
            center[0]=mask.shape[0]//2
        if np.isnan(center[1]):
            center[1]=mask.shape[1]//2

        center[0]=round(center[0])
        center[1]=round(center[1])

        data[i_data] = torch.FloatTensor(data[i_data]).permute(2, 0, 1)
        data[i_data] = cut_pad_image(data[i_data], center, cut_size, 0)

    return data


def clean_cut_single(data, cut_size=64, mask_channel=2, min_threshold=0):
    # tries to detect individual cells inside the image and only cuts out those.
    data_out=[]
    for i_data in tqdm(range(data.__len__()), desc=sys._getframe().f_code.co_name + ": "):
        mask = nsbatwm.threshold_isodata(data[i_data][:, :, mask_channel])
        mask_labels = nsbatwm.voronoi_otsu_labeling(mask, 3.0, 2.0)
        mask_labels = nsbatwm.remove_labels_on_edges(mask_labels)
        # images = []
        # print(str(np.max(mask_labels)))
        for i_mask in range(1, np.max(mask_labels) + 1):
            current_mask = (mask_labels == i_mask).astype(np.int32)
            current_mask = nsbatwm.expand_labels(current_mask, 3.0)

            mask_x, mask_y = np.where(current_mask)

            center = [0, 0]
            center[0] = np.mean(mask_x)
            center[1] = np.mean(mask_y)

            center[0] = round(center[0])
            center[1] = round(center[1])

            tmp_image = data[i_data] * current_mask[:, :, np.newaxis]

            tmp_image = torch.FloatTensor(tmp_image).permute(2, 0, 1)
            tmp_image = cut_pad_image(tmp_image, center, cut_size, 0)

            data_out.append(tmp_image)

        # fi(mask)
        # project_0 = np.sum(mask > np.quantile(mask.flatten(), 0.8), 1)
        # project_1 = np.sum(mask > np.quantile(mask.flatten(), 0.8), 0)
        #
        # center = [0, 0]
        # center[0] = np.nanmean([i for i in range(mask.shape[0]) if project_0[i] > 0])
        # center[1] = np.nanmean([i for i in range(mask.shape[1]) if project_1[i] > 0])
        #
        # if np.isnan(center[0]):
        #     center[0]=mask.shape[0]//2
        # if np.isnan(center[1]):
        #     center[1]=mask.shape[1]//2
        #
        # center[0]=round(center[0])
        # center[1]=round(center[1])
        #
        # data[i_data] = torch.FloatTensor(data[i_data]).permute(2, 0, 1)
        # data[i_data] = cut_pad_image(data[i_data], center, cut_size, 0)

    return data_out

def clean_median(data, remove_hotpixels=False):
    for i_data in tqdm(range(data.__len__()), desc=sys._getframe().f_code.co_name + ": "):
        for i_channel in range(data[i_data].shape[2]):
            #sometimes there are hot pixels (>0.8). fill those with 2d median filter
            if remove_hotpixels:
                fill = scipy.signal.medfilt2d(data[i_data][:, :, i_channel])
                mask = data[i_data][:, :, i_channel] > 0.8
                data[i_data][:, :, i_channel][mask] = fill[mask]
            median = np.median(data[i_data][:, :, i_channel].flatten())
            data[i_data][:, :, i_channel] = (data[i_data][:, :, i_channel] - median)
    return data


def clean_normalize(data, var):
    for i_data in tqdm(range(data.__len__()), desc=sys._getframe().f_code.co_name + ": "):
        for i_channel in range(data[i_data].shape[2]):
            data[i_data][:, :, i_channel] = data[i_data][:, :, i_channel]/var[i_channel]
    return data


def clean_pca_orientation(data):
    for i_data in tqdm(range(data.__len__()), desc=sys._getframe().f_code.co_name + ": "):
        mask = np.amax(np.abs(data[i_data]), 2)

        mask = mask > np.quantile(mask.flatten(), 0.9)
        x, y = mask.nonzero()

        X = [[ix, iy] for ix, iy in zip(x, y)]

        pca = PCA(n_components=2)
        pca.fit(X)
        rotation_angle = math.degrees(math.atan(pca.components_[0][0] / (pca.components_[0][1]+1e-5)))
        data[i_data] = ndimage.rotate(data[i_data], rotation_angle + 90, mode='nearest', axes=(0, 1), reshape=False)

    return data


def load_cif(filename, channels=None, outliers=[], inliers=[]):
    print('loading cif file: ' + filename)
    with bioformats.ImageReader(filename, perform_init=True) as reader:
        image_count = reader.rdr.getSeriesCount()
        if inliers:
            outliers = list(range(image_count//2))
            outliers = [o for o in outliers if o not in inliers]
        load_cif_data = list()
        for i_image in tqdm(range(0, image_count, 2), desc="load_cif: "):
            if i_image//2 in outliers:
                continue
            tmp_data = []
            for channel in channels:
                tmp_data.append(reader.read(series=i_image, c=channel))
            load_cif_data.append(np.stack(tmp_data, axis=2))

    return load_cif_data

def load_zip(filename, channels=None, outliers=[], inliers=[]):
    print('loading zip file: ' + filename)
    archive = zipfile.ZipFile(filename, 'r')
    namelist = archive.namelist()
    image_names = natsorted(list(set([Path(x).name.split("_")[0] for x in namelist])))
    folder = str(Path(namelist[0]).parent)
    if inliers:
        outliers = [int(x) for x in image_names]
        outliers = [o for o in outliers if o not in inliers]
    load_data = []
    for i_image in tqdm(image_names, desc="loading_zip: "):
        if i_image in outliers:
            continue
        tmp_data = []
        for channel in channels:
            tmp_name = os.path.join(folder, i_image + "_Ch" + str(channel) + ".ome.tif")
            # tmp_data.append(archive.open(tmp_name))
            with io.BytesIO(archive.read(tmp_name)) as file_like_object:
                with TiffFile(file_like_object) as tiff_file:
                    image = tiff_file.asarray()
                    tmp_data.append(image.astype(np.float32))
        load_data.append(np.stack(tmp_data, axis=2))

    return load_data


def read_outliers(path):
    if os.path.isfile(path):
        file = open(path, 'r')
        Lines = file.readlines()
        outliers = [int(x.strip()) for x in Lines]
    else:
        outliers=[]
    return outliers