
import os
import skimage
import skimage.io
import skimage.color
from skimage.feature import hog
from preprocessing.preprocess import custom_hog,hog_feature_arr,overlap
from detector.utils import get_config
import joblib
import numpy as np

class FeatureExtractor(object):
    """
    Class to extract training and test feature vectors from raw files
    """
    def __init__(self, config):
        """

        :param config: Dictionary of Parameters
        """
        self.stride = config['slidingStep']
        self.block_size = config['HOGBlockSize']
        self.n_bins = config['HOGBins']
        self.cell_size = config['HOGCellSize']
        self.window_width = config['slidingWindowWidth']
        self.window_height = config['slidingWindowHeight']
        self.feature_count = config['HOGFeatureCount']
        self.main_path = config['datapath']
        self.train_path = config['trainingData']
        self.test_path = config['testingData']

    def get_training_skhog_features(self):
        """
        Extracts and returns list of HOG feature vectors for the entire training set.

        :param config: Dictionary containing configuration parameters
        :return: All instances of training set in HOG descriptor form
        """

        samples = []

        for pic_name in os.listdir(self.train_path):
            img_raw = skimage.img_as_float(skimage.io.imread( self.train_path + "/" + pic_name))
            for img in (img_raw, img_raw[:, ::-1]):
                feature_arr = np.empty((self.feature_count,), dtype=np.float64)

                hog_feature_arr(custom_hog(img, self.cell_size, self.n_bins), self.block_size, self.n_bins, feature_arr)
                #feature_arr = hog(img, pixels_per_cell=(10, 10), block_size=(4, 4))
                data = {'fv': feature_arr,'class_': [0, 1] if pic_name[:3] == "pos" else [1, 0]}
                samples.append(data)

        return samples



    def get_testing_skhog_features(self):
        """
        Extracts and returns list HOG feature vectors for the entire test set.
        Sliding window approach is used to extract feature vectors. An overlap threshold is used against
        the bounding box & sliding window overlap to determine the class of the instance
        i.e. if the current sliding window has an overlap proportion >= 0.8 with an annotated bounding box,
        then the instance is considered positive.

        :return: All instances in test set in HOG descriptor form
                """

        boxes =self.get_bounding_boxes()
        samples = []

        for pic_name in os.listdir(self.test_path):
            img_index = int(pic_name.split('.')[0].split('-')[1])
            print(img_index)
            img = skimage.img_as_float(skimage.io.imread(self.test_path + "/" + pic_name))
            for y in range(0,img.shape[0]-self.window_height, self.stride):
                for x in range(0,img.shape[1]-self.window_width, self.stride):
                    xe, ye = x + self.window_width, y+self.window_height
                    window_box = np.array([(x, y), (xe, ye)])

                    feature_arr = np.empty((self.feature_count,), dtype=np.float64)
                    hog_feature_arr(custom_hog(img[y:ye, x:xe], cell_size=self.cell_size, n_bins = self.n_bins),\
                                    self.block_size, self.n_bins, feature_arr)
                    #feature_arr = hog(img[y:ye, x:xe], pixels_per_cell=(10, 10), block_size=(4, 4))
                    for bounding_box in boxes[img_index]:
                        if overlap(bounding_box,window_box) > 0.775:

                            class_ = [0, 1]
                            break
                    else:
                        class_ = [1,0]
                    data = {'fv':feature_arr, "class_":class_}
                    samples.append(data)
        return samples


    def get_bounding_boxes(self):
        """
        Parses and returns bounding box data for test set

        :param config: Dictionary containing configuration parameters
        :return: Bounding boxes of the test set

        """
        with open(self.main_path+ "/trueLocations.txt") as fin:
            top_left_coords = fin.read().split('\n')

        top_left_coords = [item.split(":")[1].strip().split(' ') for item in top_left_coords[:-1]]
        boxes = []

        for i, box in enumerate(top_left_coords):
            boxes.append([])
            top_left_coords[i] = [[int(n) for n in c.strip('()').split(',')][::-1] for c in box]
            for c in top_left_coords[i]:
                end = c[0] + self.window_width, c[1] + self.window_height
                boxes[i].append(np.array([c, end]))
        return boxes

    def get_bounding_boxes_scale(self):
        """
                Parses and returns bounding box data for multiscale test set

                :param config: Dictionary containing configuration parameters
                :return: Bounding boxes of the test set

        """
        with open(self.main_path + "/trueLocations_Scale.txt") as fin:
            top_left_coords = fin.read().strip().split('\n')

        top_left_coords = [item.split(":")[1].strip().split(' ') for item in top_left_coords]
        boxes = []

        for i, box in enumerate(top_left_coords):
            boxes.append([])
            top_left_coords[i] = [[int(n) for n in c.strip('()').split(',')] for c in box]
            for c in top_left_coords[i]:
                end = c[0] + self.window_width, c[1] + self.window_height
                boxes[i].append(c)
        return boxes

if __name__ == '__main__':
    config = get_config()
    if not os.path.isdir(config['datapath'] + "/feature"):
        os.mkdir(config['datapath'] + "/feature")
    extractor = FeatureExtractor(config)
    joblib.dump([extractor.get_training_skhog_features()], config['datapath'] + "/feature/trainingData.pkl",compress=1)
    joblib.dump([extractor.get_testing_skhog_features()], config['datapath'] + "/feature/testingData.pkl",compress=1)