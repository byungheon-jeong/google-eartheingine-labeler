import os,time, pickle, re
from tkinter.messagebox import NO
from unittest.case import TestCase
import rasterio #for reading images
import napari, unittest

import multiprocessing as mlt
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from matplotlib import path
from scipy.sparse import data
from argparse import ArgumentParser

directory = os.path.join(os.getcwd(), "ee_data")

def runNapari(img_path, full_img_path):
    with rasterio.open(img_path) as src:
        img = src.read()
    
    with rasterio.open(full_img_path) as src:
        img_full = src.read()  

    viewer = napari.view_image(img)
    return viewer, img, img_full

def getPolygonMasks(viewer):
    layers = [layer for layer in viewer.layers]
    label_data = {}

    for layer in layers[1:]:
        name = layer.name
        layer_type = re.search(r"([A-Za-z ]*)\s?[0-9]?", name).group(0).strip()
        label_data.setdefault(layer_type, []).append(np.delete(layer.data[0],(0),axis=1))

    return label_data

def containsWithin(path_dimention, img):
    glacier = path.Path(path_dimention)
    indices = np.where(np.all(img == img, axis=0))
    pixels = np.array(list(zip(indices[0],indices[1])))
#     return pixels
    return glacier.contains_points(pixels).reshape(img.shape[1:])


def getPixelMask(img, paths):

    all_masks = {}
    all_coordinates = {}
    mask_labels = paths.keys()
    img_dimentions = np.rollaxis(img,0,3)[0,0,:].shape

    for label in mask_labels:
        partial_coordinates = np.where(np.logical_or.reduce([containsWithin(path,img) for path in paths[label]]))
        coordinates = list(zip(partial_coordinates[0], partial_coordinates[1]))
        all_coordinates[label] = coordinates
        # mask = np.array([], ndmin=img.shape[0])
        # mask = np.empty(shape=np.rollaxis(img,0,3)[0,0,:].shape)
        mask = []
        for (x,y) in coordinates:
            img_coordinate_values = np.rollaxis(img,0,3)[x,y,:]
            mask.append(img_coordinate_values)
            # mask = np.append(mask,np.rollaxis(img,0,3)[x,y,:])

        all_masks[label] = np.array(mask)
        all_coordinates[label] = np.array(coordinates)

    return all_masks, all_coordinates, img_dimentions


def getTrainingData(masks,img_dimentions):
    data = np.empty(shape=img_dimentions)
    labels = np.array([])
    for key, mask in masks.items():
        label = np.array([key]*mask.shape[0])
        labels = np.append(labels,label)
        data = np.vstack((data,mask))
        # np.vstack((labels, label))
        # np.hstack((data, mask))

    return data[1:], labels
    # data,labels = np.array([]), np.array([])

    # iceLabels = np.array(["Ice"]*iceData.shape[0])
    # nonIceLabels = np.array(["Not Ice"]*nonIceData.shape[0])
    # data = np.vstack((iceData, nonIceData))
    # labels = np.hstack((iceLabels, nonIceLabels))

    # return data, labels


def loadCheckpoint(directory,num_bands):
    logpath = os.path.join(directory,"checkpoints", "imgAnnotatedData.npy")
    try:
        with open(logpath, "rb") as f:
            imageList = np.load(f)

    except (OSError, IOError) as e:
        imageList = np.array([])
        try :
            os.mkdir(os.path.join(directory,"checkpoints"))
        except:
            print("Data dir is present")


    data, labels = np.empty([0,num_bands]),np.empty([0])
    
    if os.path.exists(os.path.join(directory, "checkpoints","data.npy")) and os.path.exists(os.path.join(directory, "checkpoints","labels.npy")):
        with open(os.path.join(directory, "checkpoints", "data.npy"),'rb') as f:
            data= np.load(f)
        with open(os.path.join(directory, "checkpoints", "labels.npy"),'rb') as f:
            labels=np.load(f)

    return imageList, data, labels


def updateLog(imageList,directory):
    logpath = os.path.join(directory, "imgAnnotatedData.npy")
    with open(logpath, "wb") as f:
        np.save(f, imageList)


def updateArraysAndSave(data,image_data,labels,image_labels,directory):
    data = np.vstack((data,image_data))
    labels = np.hstack((labels, image_labels))

    with open(os.path.join(directory, "data.npy"),"wb") as f:
        np.save(f, data)
    with open(os.path.join(directory, "labels.npy"), "wb") as f:
        np.save(f, labels)


def runTestsAndLog(img, coordiantes, paths, viewer, annotated_list,image_file,checkpoint_directory):
    testPixelMask(img, coordiantes, paths, viewer)
    annotated_list = np.append(annotated_list, image_file)
    updateLog(annotated_list, checkpoint_directory)
    input("Input ENTER after checking RB pixels")


def testPixelMask(img,coordiantes,paths,viewer):
    # iceMask = np.where(np.logical_or.reduce([containsWithin(path,img) for path in paths["ice"]]))
    # nonIceMask = np.where(np.logical_or.reduce([containsWithin(path,img) for path in paths["non_ice"]]))
    
    for label, coordinate in coordiantes.items():
        colors = np.random.randint(0,255,size=(3,))
        viewer.add_points(coordinate,face_color=colors,edge_color =colors,size=10, name=f"{label}_points")

    # ice_coordinates = list(zip(iceMask[0],iceMask[1]))
    # non_ice_coordinates = list(zip(nonIceMask[0],nonIceMask[1]))

    # viewer.add_points(ice_coordinates,face_color="red",edge_color ="red",size=1, name="Ice")
    # viewer.add_points(non_ice_coordinates,face_color="blue",edge_color ="blue",size=1, name="Not Ice")

    return None

# def test():
#     try:
#         test_image_path = input("Input the image to test:\n")
#         with rasterio.open(test_image_path) as src:
#             img = src.read()
#     except:
#         raise ValueError("YOU MUST INPUT PATH TO .TIF IMAGE")
#     test_viewer, = runNapari(img, img)
#     # napari_process = mlt.Process(target=napari.run())
#     # napari_process.run()xs

#     while True:
#         input("Press Enter after Labeling")
#         try:    
#             paths = getPolygonMasks(test_viewer)
#             test_mask = getPixelMask(img,paths)
#             testPixelMask(img, paths, test_viewer)
#             break
#         except:
#             print("Label the image")
#     return test_mask


def main():
    directory = input("Enter Directory:\n")
    image_directory = os.path.join(directory,"ndsi_img")
    full_data_directory = os.path.join(directory, "full_img")
    checkpointdirectory = os.path.join(directory, "checkpoints")
    # directory = r"C:\Users\marke\Documents\DSC180A-Q1\ee_data"
    num_bands = rasterio.open(os.path.join(full_data_directory,list(os.walk(full_data_directory))[0][-1][-1])).read().shape[0]


    annotated_list, data, labels = loadCheckpoint(directory,num_bands)

    for root, dirs, files in os.walk(image_directory):
        for image_file in files:
            image_path = os.path.join(image_directory, image_file)
            full_image_path = os.path.join(full_data_directory, image_file)
            print(image_path)
            if os.path.splitext(image_path)[1] == ".tif" and image_file not in annotated_list:
                viewer,img, img_full = runNapari(image_path,full_image_path)
                while True:
                    response = input("Press Enter after Labeling or input \"SKIP\" in order to skip image:\n")
                    if response == "SKIP":
                        viewer.close()
                        annotated_list = np.append(annotated_list, image_path)
                        updateLog(annotated_list, checkpointdirectory)
                        break
                    try:    
                        paths = getPolygonMasks(viewer)
                        # To be changed
                        masks, coordinates, img_dimentions = getPixelMask(img_full,paths)
                        image_data,image_labels = getTrainingData(masks,img_dimentions)
                        runTestsAndLog(img, coordinates, paths, viewer, annotated_list,image_file,checkpointdirectory)
                        updateArraysAndSave(data,image_data,labels,image_labels,checkpointdirectory)

                        viewer.close()
                        break
                    except Exception as e:
                        print(f"{e} \nLabel the image")

                # return test_mask
        

if __name__ == "__main__":

    main()