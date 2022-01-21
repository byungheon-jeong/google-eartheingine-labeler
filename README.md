# google-eartheingine-labeler

This is the project repo for a python labeling script that uses napari (https://github.com/napari/napari) to label a google earth engine dataset.


- Installation (Conda)
  - conda create --name <name> --file ./requirements.txt

- Dataset
  - label_img: images with less than 3 dimentions that will be used to label. 
  - full_img: images with all the bands that is to be used in the dataset

- Directions
  > 
  python .\src\run_annotation.py 
  >
  
  Enter Directory: <Location of Dataset Directory with label_img & full_img subdirectories
  
  Label Image or SKIP
  
  
  
  

