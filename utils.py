import os
import ujson as json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split


def get_data(config):
    data = []
    for index,feature_file in enumerate(os.listdir(config["datapath"]+"/feature")):
        instance = joblib.load(config["datapath"]+"/feature/"+feature_file)
        data.append(np.hstack((instance['fv'], np.array([instance['label']]))))
        print(index)
    return data
def split_training_test(feature_file):
    data = joblib.load(feature_file)

    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1337)
    joblib.dump({'x_train': X_train, 'y_train': y_train}, 'K:/Master/MachineLearning/dataset4/dataset4_train.pkl')
    joblib.dump({'x_test': X_test, 'y_test': y_test},'K:/Master/MachineLearning/dataset4/dataset4_test.pkl')

def get_training_data(config):
    data = joblib.load(config['trainingData'])
    return data['x_train'], data['y_train']

def get_testing_data(config):
    data = joblib.load(config['testingData'])
    return data['x_test'], data['y_test']

def get_config():
    with open("config.json") as fin:
        config = json.load(fin)
    return config
def pack_features(output_path):
    joblib.dump(np.array(get_data(get_config())), output_path)
if __name__ == '__main__':

    feature_file = "K:/Master/MachineLearning/dataset4/big_feature/dataset4.pkl"
    joblib.dump(np.array(get_data(get_config())),feature_file)

    split_training_test(feature_file)
    print("done")