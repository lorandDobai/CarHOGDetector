import numpy as np
import joblib
import sklearn.utils
import os
from sklearn.metrics import accuracy_score,classification_report
import skimage
import skimage.io
import skimage.transform
from detector import utils
from preprocessing import feature_extract
from preprocessing.preprocess import display,plot_bounding_box,get_hog_feature_array
from neural_network import learning_utils
from neural_network.neural_network2 import MLPClassifier


def train_mlp_model(config):
    """
    Trains the model using the preprocessed training set.

    :param config: Dictionary of parameters
    :return: None
    """
    x,y = utils.get_training_data(config)
    x, y = sklearn.utils.shuffle(x, y)

    clf = MLPClassifier(layers=(x.shape[1],8, 2), activation='relu', learning_rate=0.001, regularization=1e-2,
                        batch_size=32, training_epochs=500)
    clf.fit(x,learning_utils.one_hot_encode(y))
    return clf

def validate_mlp_model(model):
    """
    Evaluates the performance of the model against the standard test dataset. Prints a classification report which
    contains measurements for each class.
    :param model:  Trained MLP Model
    :return: None
    """
    x_test, y_test = utils.get_testing_data(utils.get_config())
    #y_pred = model.decision_function(x_test)[:,1]>=0.825
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))
    trues = y_test == 1
    pos_x_test, pos_y_test = x_test[trues], y_test[trues]
    pos_y_pred = model.predict(pos_x_test)
    print(accuracy_score(pos_y_test, pos_y_pred))


def preview_model(model):
    """
    Iterate through every file in test set and plot system prediction.
    To help reduce clutter from multiple overlapping positive prediction, a non-maximum suppresion algorithm is used.
    Also, a candidate is considered to be positive only if it's decision function probabilty exceeds a threshold (0.95)

    :param model: Trained MLP model
    :return: None
    """
    # Parse annotation file
    bounding_boxes =  feature_extract.FeatureExtractor(utils.get_config()).get_bounding_boxes_scale()
    config = utils.get_config()
    data_path = config['scaleTestingData']
    w,h = config['slidingWindowWidth'], config['slidingWindowHeight']
    # Iterate through every file in test set
    for pic_name in os.listdir(data_path):
        # Extract index from file name
        index = int(pic_name.split('.')[0].split('-')[1])

        # read image
        image = skimage.io.imread("{}/{}".format(data_path,pic_name))
        scales = {}
        candidates = []
        # iterate through each annotation
        for annot_box in bounding_boxes[index]:
            scaling = w/annot_box[2]
            #   rescale image to annotation scale
            scaled_image = skimage.transform.rescale(image,scaling)

            # search for cars using a sliding window approach

            for y in range(0,scaled_image.shape[0]-h,4):
                for x in range(0, scaled_image.shape[1]-w, 4):
                    # extract hog features
                    fv = get_hog_feature_array(scaled_image[y:y+h,x:x+w],config['HOGFeatureCount'],\
                                                config['HOGCellSize'],config['HOGBlockSize'], config['HOGBins'])
                    # retain only those who have a decision function probability higher than a threshold
                    if(model.decision_function(np.array([fv]))[0][1]>=0.9):
                        rscale = 1/scaling
                        candidates.append(np.array(list(map(lambda x: int(x),
                                                            [x*rscale,y*rscale,(x+w)*rscale,(y+h)*rscale]))))
                        scales[tuple(candidates[-1][:3])] = rscale
            cars = []
            # apply non-maximum suppresion
        for c in utils.non_max_suppression_fast(np.array(candidates),0.375):
            x,y = c[:2]
            scale = scales[tuple(c[:3])]
            cars.append([(x,y),(x,c[3]),(c[2],c[3]),(c[2],y)])
        plot_bounding_box(image, cars)


if __name__ == '__main__':
    #model = train_mlp_model(utils.get_config())
    #joblib.dump(model, "models/car_detector.pkl")
    model = joblib.load("models/car_detector.pkl")

    #utils.pr_curve(model)
    validate_mlp_model(model)

    preview_model(model)