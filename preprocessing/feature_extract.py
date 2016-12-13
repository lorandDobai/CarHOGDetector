import ujson as json
import os
import skimage.io
from skimage.feature import hog
from preprocessing.common_utils import custom_hog,calculate_feature_array
from skimage.color import rgb2gray
import joblib
import numpy as np
from skimage import img_as_float
def extract_skhog_features(config):
    if not os.path.isdir(config['datapath']+"/feature"):
        os.mkdir(config['datapath']+"/feature")
    for class_ in ('/positive/','/negative/'):
        for pic_name in os.listdir(config['datapath']+class_) :#+ os.listdir(config['datapath']+'/negative'):
            img = rgb2gray(skimage.io.imread(config['datapath']+class_+pic_name))

            fd = hog(img,pixels_per_cell=(16,16),cells_per_block=(4,4))
            with open(config['datapath']+'/annotations_labels/'+pic_name[:-4]+'.json') as fin:
                label = json.load(fin)
            data = {'fv':fd}
            data.update(label)

            joblib.dump(data,config['datapath']+"/feature/"+pic_name[:-4]+".pkl",compress=True)

def extract_myhog_features(config):
    if not os.path.isdir(config['datapath']+"/feature"):
        os.mkdir(config['datapath']+"/feature")
    for class_ in ('/positive/','/negative/'):
        for pic_name in os.listdir(config['datapath']+class_) :#+ os.listdir(config['datapath']+'/negative'):
            img = img_as_float(skimage.io.imread(config['datapath']+class_+pic_name))

            fd = custom_hog(img,cell_size=16)
            out_arr = np.zeros((1728,),dtype=np.float64)
            calculate_feature_array(fd,4,9,out_arr)
            with open(config['datapath']+'/annotations_labels/'+pic_name[:-4]+'.json') as fin:
                label = json.load(fin)
            data = {'fv':out_arr}
            data.update(label)

            joblib.dump(data,config['datapath']+"/feature/"+pic_name[:-4]+".pkl",compress=True)
if __name__ == '__main__':
    with open("../config.json") as fin:
        config = json.load(fin)
        extract_myhog_features(config)