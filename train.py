import os, rasterio
import numpy as np
from src.run_annotation import loadCheckpoint, runNapari, updateLog, getPolygonMasks, getPixelMask, getTrainingData, runTestsAndLog, updateArraysAndSave

def main():
    directory = os.path.join(os.getcwd(), "testing_data")
    image_directory = os.path.join(directory,"label_img")
    full_data_directory = os.path.join(directory, "full_img")
    checkpointdirectory = os.path.join(directory, "checkpoints")
    num_bands = rasterio.open(os.path.join(full_data_directory,list(os.walk(full_data_directory))[0][-1][-1])).read().shape[0]
    annotated_list, data, labels = loadCheckpoint(directory,num_bands)

    # Iterate through the images in directory
    for root, dirs, files in os.walk(image_directory):
        for image_file in files:
            
            annotated_list, data, labels = loadCheckpoint(directory,num_bands)
            # Get the image that will be used for labeling
            image_path = os.path.join(image_directory, image_file)
            # Get the images that have all the bands
            full_image_path = os.path.join(full_data_directory, image_file)
            print(image_path)
            if os.path.splitext(image_path)[1] == ".tif" and image_file not in annotated_list:
                viewer,img, img_full = runNapari(image_path,full_image_path)
                while True:
                    response = input("Press Enter after Labeling or input \"SKIP\" in order to skip image:\n")\
                    # If the TIFF image is too distorted or otherwise unusable
                    if response == "SKIP":
                        viewer.close()
                        annotated_list = np.append(annotated_list, image_path)
                        updateLog(annotated_list, checkpointdirectory)
                        break
                    try:    
                        paths = getPolygonMasks(viewer)
                        # Run the cycle per image
                        masks, coordinates, img_dimentions = getPixelMask(img_full,paths)
                        image_data,image_labels = getTrainingData(masks,img_dimentions)
                        runTestsAndLog(coordinates, viewer, annotated_list,image_file,checkpointdirectory)
                        updateArraysAndSave(data,image_data,labels,image_labels,checkpointdirectory)

                        viewer.close()
                        break
                    except Exception as e:
                        # If there was an accidential button process or some other error
                        print(f"{e} \nLabel the image")

        

if __name__ == "__main__":

    main()