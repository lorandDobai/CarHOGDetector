import os
import ujson as json
import joblib
import numpy as np
from neural_network import learning_utils
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
def get_training_data(config):
    data = joblib.load(config['datapath'] + "/feature/trainingData.pkl")[0]
    x = []
    y = []
    for sample in data:
        x.append(sample['fv'])
        y.append(sample['class_'].index(1))
    return np.array(x), np.array(y)

def get_testing_data(config):
    data = joblib.load(config['datapath'] + "/feature/testingData.pkl")[0]
    x = []
    y = []
    for sample in data:
        x.append(sample['fv'])
        y.append(sample['class_'].index(1))
    return np.array(x), np.array(y)


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def pr_curve(model):
    precision = dict()
    recall = dict()
    average_precision = dict()
    x_test, y_test = get_testing_data(get_config())
    y_test = learning_utils.one_hot_encode(y_test)
    y_score = model.decision_function(x_test)
    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")

    plot_precision_recall(precision,recall,average_precision)

def plot_precision_recall(precision,recall,average_precision):
# Plot Precision-Recall curve

    # Plot Precision-Recall curve for each class
    plt.clf()

    for i, color in zip([1], 'r'):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-recall curve  (area = {0:0.3f})'
                       ''.format(average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def get_config():
    with open("../detector/config.json") as fin:
        config = json.load(fin)
        window_width = config['slidingWindowWidth']
        window_height = config['slidingWindowHeight']
        cell_size =  config["HOGCellSize"]
        block_size = config["HOGBlockSize"]
        n_bins = config["HOGBins"]
        config['HOGFeatureCount'] = (window_width // cell_size - block_size + 1) * \
                        (window_height //  cell_size -  block_size  + 1) * n_bins * (block_size ** 2)

    return config
