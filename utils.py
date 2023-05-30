import os
import sys
import numpy as np
import xml.etree.ElementTree as xmlET
import cv2
from torch import cuda
import gc, time

def free_memory(sleep_time=0.1):
    gc.collect()
    cuda.synchronize()
    gc.collect()
    cuda.empty_cache()
    time.sleep(sleep_time)
	
def load_image(filename):

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(img, shape, interpolation=cv2.INTER_CUBIC):

    # resize relevant image axis to length of corresponding target axis while preserving aspect ratio
    axis = 0 if float(shape[0]) / float(img.shape[0]) > float(shape[1]) / float(img.shape[1]) else 1
    factor = float(shape[axis]) / float(img.shape[axis])
    img = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=interpolation)

    # crop other image axis to match target shape
    center = img.shape[int(not axis)] / 2.0
    step = shape[int(not axis)] / 2.0
    left = int(center-step)
    right = int(center+step)
    if axis == 0:
        img = img[:, left:right]
    else:
        img = img[left:right, :]

    return img
    
def parse_convert_xml(conversion_file_path):

    defRoot = xmlET.parse(conversion_file_path).getroot()

    one_hot_palette = []
    class_list = []
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        if to_class in class_list:
             one_hot_palette[class_list.index(to_class)].append(from_color)
        else:
            one_hot_palette.append([from_color])
            class_list.append(to_class)

    return one_hot_palette
	
def one_hot_encode_image(image, palette):

    one_hot_map = []

    # find instances of class colors and append layer to one-hot-map
    for class_colors in palette:
        class_map = np.zeros(image.shape[0:2], dtype=bool)
        for color in class_colors:
            class_map = class_map | (image == color).all(axis=-1)
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = np.stack(one_hot_map, axis=-3)
    one_hot_map = one_hot_map.astype(np.float32)

    return one_hot_map
	
def one_hot_decode_image(one_hot_image, palette):

    # create empty image with correct dimensions
    height, width = one_hot_image.shape[1:3]
    depth = palette[0][0].size
    image = np.zeros([height, width, depth])

    # reduce all layers of one-hot-encoding to one layer with indices of the classes
    map_of_classes = one_hot_image.argmax(0)

    for idx, class_colors in enumerate(palette):
        # fill image with corresponding class colors
        image[np.where(map_of_classes == idx)] = class_colors[0]

    image = image.astype(np.uint8)

    return image