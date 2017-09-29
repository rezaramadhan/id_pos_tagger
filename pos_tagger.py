#!/usr/bin/python
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

import pprint
import numpy

RAW_DATA = []
DATA_FILENAME = "id-ud-train.conllu"

def load_raw_data(filename):
    # load data from file and put it in RAW_DATA
    # RAW_DATA is a list of line of token.
    # this function also removes comment starting with # and empty line from file

    global RAW_DATA

    with open(filename, "r") as fin:
        for line in fin:
            if (line[0] != '#'):
                line = line.strip('\n')
                row_data = line.split('\t')
                if len(row_data) != 1:
                    RAW_DATA.append(row_data)

def get_features(data, idx):
    # convert data to a dictionary of features. data contains raw data from file
    # and idx is the line of data being processed
    return ({
        'word': data[idx][1].lower(),
        'word-1': '_' if int(data[idx][0]) <= 1 else data[idx - 1][1].lower(),
        'word+1': '_' if (idx >= len(data) - 1) or (int(data[idx + 1][0]) == 1) else data[idx + 1][1].lower(),
        'pos-1': '_' if int(data[idx][0]) <= 1 else data[idx - 1][3],
        'pos-2': '_' if int(data[idx][0]) <= 2 else data[idx - 2][3],
        'is_first': data[idx][0] == 1,
        'is_last': (idx == len(data) - 1) or (int(data[idx + 1][0]) == 1),
        'is_numeric': data[idx][1].isdigit(),
        'prefix_2': data[idx][1][:2],
        'prefix_3': data[idx][1][:3],
        'suffix_2': data[idx][1][len(data[idx][1]) - 2:2],
        'suffix_3': data[idx][1][len(data[idx][1]) - 2:3]
    }, data[idx][3])

def reformat_data(data):
    # convert data from RAW_DATA to a formatted version.
    # the formatted data is a dictionary with a DataFrame features and a list of token class

    rows = []
    feature_rows = []
    class_rows = []
    for i in range(len(data)):
    # for i in range(25):
        (features, pos_class) = get_features(data, i)
        feature_rows.append(features)
        class_rows.append(pos_class)

    features_dframe = DataFrame(feature_rows)

    return {'features': features_dframe, 'class': class_rows}

def encode_data(data):
    # convert data from string format to integer using LabelEncoder

    labelencoder = preprocessing.LabelEncoder()
    classencoder = preprocessing.LabelEncoder()

    first_match = [s for s in data['features'].columns if "pos" in s][0]
    classencoder.fit(data['features'][first_match])
    data['class'] = numpy.array(classencoder.transform(data['class']))

    for column in data['features'].columns:
        if 'pos' in column:
            data['features'][column] = classencoder.transform(data['features'][column].values)
        else:
            labelencoder.fit(data['features'][column].values)
            data['features'][column] = labelencoder.transform(data['features'][column].values)

    return data

if __name__ == '__main__':
    load_raw_data(DATA_FILENAME)
    data = reformat_data(RAW_DATA)
    data = encode_data(data)

    print data

    pipeline = Pipeline([
        ('classifier', DecisionTreeClassifier(criterion='entropy')) # ~ 8.2
        # ('classifier', RandomForestClassifier(n_estimators=250)) # ~ 7,5
        # ('classifier', GaussianProcessClassifier()) # ~
        # ('classifier', MLPClassifier()) # ~ 7.0 -> lama banget -.-
        # ('classifier', GaussianNB()) # ~ 4.2
    ])


    k_fold = KFold(n_splits=10)
    scores = []
    for train_indices, test_indices in k_fold.split(data['class']):
        train_class = numpy.take(data['class'], train_indices)
        test_class = numpy.take(data['class'], test_indices)

        train_features = data['features'].iloc[train_indices].values
        test_features = data['features'].iloc[test_indices].values

        pipeline.fit(train_features, train_class)
        score = pipeline.score(test_features, test_class)
        print score
        scores.append(score)

    print 'Score:', sum(scores)/len(scores)
