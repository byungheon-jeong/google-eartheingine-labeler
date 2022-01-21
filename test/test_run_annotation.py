import yaml, unittest, os, re
import numpy as np
from run_annotation import containsWithin, runNapari, getPolygonMasks, getPixelMask, getTrainingData


class Test_MultiLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_file = r"C:\Users\marke\Documents\DSC180A-Q1\config\test.yml"
        with open(config_file) as config_file:
            cfg = yaml.load(config_file,Loader=yaml.Loader)
        image_path = cfg["image_path"]
        full_image_path = cfg["full_image_path"]
        viewer,img, img_full = runNapari(image_path,full_image_path)

        cls.img_full = img_full
        cls.viewer = viewer

        while True:
            response = input("Press Enter after Labeling or input \"SKIP\" in order to skip image:\n")
            if response == "SKIP":
                print("WHAT?")
            try:
                cls.viewer = viewer
                return None
            except Exception as e:
                print(e)

    def setUp(self) -> None:
        self.viewer = Test_MultiLayers.viewer
        self.img_full = Test_MultiLayers.img_full


    def test_getPolygonMasks(self):
        # MAKE SURE TO HAVE ICE AND NON ICE LABELS
        def getPolygonMasks_old(viewer)-> list:
            ice_layers = {str.lower(layer.name):layer.data for layer in viewer.layers if re.match("^ice.*", str.lower(layer.name))}
            non_ice_layers= {str.lower(layer.name):layer.data for layer in viewer.layers if re.match("^not.*ice.*", str.lower(layer.name))}
            ice_coordinate_list, non_ice_coordinate_list = list(),list()

            for i, (x,y) in enumerate(ice_layers.items()):
                ice_coordinate_list.append(np.delete(y[0],(0),axis=1))
            for i, (x,y) in enumerate(non_ice_layers.items()):
                non_ice_coordinate_list.append(np.delete(y[0],(0),axis=1))
            path_results = {"ice":ice_coordinate_list, "non_ice":non_ice_coordinate_list}
            return path_results

        old_paths = getPolygonMasks_old(self.viewer)
        new_paths = getPolygonMasks(self.viewer)
            
        self.assertEqual(old_paths, new_paths)


    def test_get_training_data(self):
        paths = getPolygonMasks(self.viewer)
        masks,img_dimentions = getPixelMask(self.img_full,paths)

        data, labels = getTrainingData(masks,img_dimentions)
        print(data.shape, labels.shape)

        
    def test_getPixelMask(self):

        def getPixelMask_old(img,paths):
            iceMask = np.where(np.logical_or.reduce([containsWithin(path,img) for path in paths["ice"]]))
            nonIceMask = np.where(np.logical_or.reduce([containsWithin(path,img) for path in paths["non_ice"]]))

            ice_coordinates = list(zip(iceMask[0],iceMask[1]))
            non_ice_coordinates = list(zip(nonIceMask[0],nonIceMask[1]))    
            iceData,nonIceData=list(),list()
            for (x,y) in ice_coordinates:
                iceData.append(np.rollaxis(img,0,3)[x,y,:])
            for (x,y) in non_ice_coordinates:
                nonIceData.append(np.rollaxis(img,0,3)[x,y,:])
            return np.array(iceData), np.array(nonIceData)

        paths = getPolygonMasks(self.viewer)
        old_masks = getPixelMask_old(self.img_full, paths)

        masks = getPixelMask(self.img_full,paths)


if __name__ == "__main__":
    unittest.main()

    # with open(os., "wb") as f: pickle.dump(layers, f)