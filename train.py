from copy import deepcopy
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import utils
import joblib
from neural_network.neural_network2 import SimpleMLPClassifier
from neural_network import learning_utils
def train_mlp_model(config):
    print("Training MLP...")
    scores = []
    X, y = utils.get_training_data(config)

    skf = StratifiedKFold(n_splits=10)
    max_score = 0
    best_clf = None
    for train, test in skf.split(X, y):
        clf = SimpleMLPClassifier(input_size=1728, learning_rate=0.01, hidden_layer_size=8, regularization=1e-5,
                                  batch_size=16, training_epochs=10)
        X_train, y_train = X[train], y[train]
        clf.fit(X_train, learning_utils.process_class_representation(y_train))


        X_test, y_test = X[test], y[test]
        y_pred = clf.predict(X_test)
        print("Accuracy: {:3f} F1: {:3f}".format(accuracy_score(y_test,y_pred), f1_score(y_test,y_pred)))
        f1 = f1_score(y_test, y_pred)
        scores.append(f1)
        if f1 > max_score:
            best_clf = deepcopy(clf)
            max_score = f1

    scores = np.array(scores)
    print("Cross Validation F1_Score:", np.mean(scores), "(+/-", np.std(scores)*2, ")")
    X, y = utils.get_training_data(config)
    y_pred = best_clf.predict(X)
    print("Test F1:", f1_score(y, y_pred))
    print("Test Accuracy ", accuracy_score(y, y_pred))
    joblib.dump(best_clf,config['savePath']+"/mlp_model.pkl")

if __name__ == '__main__':
    train_mlp_model(utils.get_config())