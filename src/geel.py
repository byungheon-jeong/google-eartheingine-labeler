import os, re, json
import rasterio #for reading images
import napari
import numpy as np
from matplotlib import path

def runNapari(img_path, full_img_path):
    """
    Opens the napari UI

    Args:
        img_path (str): [path to images files that will be used for labeling]
        full_img_path (str): [path to TIFF files that contain all bands

    Returns:
        viewer: the Napari API
    """
    with rasterio.open(img_path) as src:
        img = src.read()
    
    with rasterio.open(full_img_path) as src:
        img_full = src.read()  

    viewer = napari.view_image(img)
    return viewer, img, img_full


def getPolygonMasks(viewer):
    """
    Gets the coordinates of the edges of drawn polygon from Napari
    Args:
        viewer (napari API): napari API

    Returns:
        label_data: coordinates
    """
    layers = [layer for layer in viewer.layers]
    label_data = {}

    for layer in layers[1:]:
        name = layer.name
        # layer_type = re.search(r"([A-Za-z_ ]*)\s?[0-9]?", name).group(0).strip()
        label_data.setdefault(name, []).append(np.delete(layer.data[0],(0),axis=1))

    return label_data


def containsWithin(path_dimention, img):
    """
    Gets the coordinates of all points within the polygon specified by the path_dimention

    Args:
        path_dimention (list): coordinates of polygon shapes
        img (TIFF): image to label

    Returns:
        np.array: contains pixels that are in polygon
    """
    polygon = path.Path(path_dimention)
    # These arrays are building a pixel map of image
    indices = np.where(np.all(img == img, axis=0))
    pixels = np.array(list(zip(indices[0],indices[1])))
    # return picked pixel coordinates 
    return polygon.contains_points(pixels).reshape(img.shape[1:])


def getPixelMask(img, paths):
    """
    This Gets the coordinate list of all polygons specified in paths

    Args:
        img (TIFF): image to be used during labeling
        paths np.array: coordiantes of paths

    """

    all_masks = {}
    all_coordinates = {}
    mask_labels = paths.keys()
    img_dimentions = np.rollaxis(img,0,3)[0,0,:].shape

    for label in mask_labels:
        partial_coordinates = np.where(np.logical_or.reduce([containsWithin(path,img) for path in paths[label]]))
        coordinates = list(zip(partial_coordinates[0], partial_coordinates[1]))
        all_coordinates[label] = coordinates

        mask = []
        for (x,y) in coordinates:
            img_coordinate_values = np.rollaxis(img,0,3)[x,y,:]
            mask.append(img_coordinate_values)
            # mask = np.append(mask,np.rollaxis(img,0,3)[x,y,:])

        all_masks[label] = np.array(mask)
        all_coordinates[label] = np.array(coordinates)

    return all_masks, all_coordinates, img_dimentions


def getTrainingData(masks,img_dimentions):
    """
    Generation of labels from aggregated polygon paths

    Args:
        masks (np.array): cointains all the mask coordinates
        img_dimentions (np.array): dimnetions of image H,W

    Returns:
        np.array: indexed starting from 1 to avoid created value at index 0
    """
    data = np.empty(shape=img_dimentions)
    labels = np.array([])
    unique_labels = set([])

    for key, mask in masks.items():
        # Make labels for polygon
        key_name = str(re.match(r"([a-zA-Z_0-9 ]*)([\[]?)", key).group(1)).strip()
        label = np.array([key_name]*mask.shape[0])
        labels = np.append(labels,label)
        data = np.vstack((data,mask))

    return data[1:], labels


def loadCheckpoint(directory,num_bands):
    """
    Implements Checkpoint system by reading list of images already read

    Args:
        directory (str): path of directory
        num_bands (int): number of bands within full TIFF
    """
    logpath = os.path.join(directory,"checkpoints", "imgAnnotatedData.json")
    imageList = set([])
    try:
        with open(logpath, "r") as f:
            imageList = json.loads(f.read())
            imageList = set(imageList)

    except (OSError, IOError) as e:
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
    """
    Updates the checkpoint image list

    Args:
        imageList (list): contains all images
        directory (str): path to the directory with script and checkpoint files
    """
    logpath = os.path.join(directory, "imgAnnotatedData.json")
    imageList = list(imageList)
    with open(logpath, "w") as f:
        f.write(json.dumps(imageList))


def updateArraysAndSave(data,image_data,labels,image_labels,directory):
    """
    Saves data from current image

    Args:
        data (np.array): Aggregation of Previous image data
        image_data (np.array): Current image data
        labels (np.array): Aggregation of Previous image labels
        image_labels (np.array): Current image labels 
        directory (str): Path to directory with the save files
    """
    data = np.vstack((data,image_data))
    labels = np.hstack((labels, image_labels))

    with open(os.path.join(directory, "data.npy"),"wb") as f:
        np.save(f, data)
    with open(os.path.join(directory, "labels.npy"), "wb") as f:
        np.save(f, labels)


def runTests(coordiantes, viewer):
    """
     Waits for user input to cycle through images

    Args:
        coordiantes (np.array): Coordinates of all pixels in the polygons
        viewer (napari API): 
        annotated_list (np.array): List with all the previous images
        image_file (str): Name of current TIFF image
        checkpoint_directory (str): Directory of checkpoint files
    """
    testPixelMask(coordiantes, viewer)
    input("Input ENTER after checking RB pixels")


def testPixelMask(coordiantes,viewer):
    """
    Prints labeled polygon using points so that user can see if their labels are represented correctly

    Args:
        coordinates (np.array): All the coordinates in listed polygons
        viewer (napari API): 
    """
    
    for label, coordinate in coordiantes.items():
        colors = np.random.randint(100,255,size=(3,))
        viewer.add_points(coordinate,face_color=colors,edge_color =colors,size=10, name=f"{label}_points")

    return None
