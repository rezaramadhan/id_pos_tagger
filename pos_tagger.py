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

import pprint
import numpy

RAW_DATA = []
DATA_FILENAME = "id-ud-dev.conllu"

def load_raw_data(filename):
    global RAW_DATA

    with open(filename, "r") as fin:
        for line in fin:
            if (line[0] != '#'):
                line = line.strip('\n')
                row_data = line.split('\t')
                if len(row_data) != 1:
                    RAW_DATA.append(row_data)

def get_features(data, idx):
    return ({
        'word': data[idx][1].lower(),
        'word-1': '_' if int(data[idx][0]) <= 1 else data[idx - 1][1].lower(),
        'word+1': '_' if (idx >= len(data) - 1) or (int(data[idx + 1][0]) == 1) else data[idx + 1][1].lower(),
        'pos-1': '_' if int(data[idx][0]) <= 1 else data[idx - 1][3],
        'pos-2': '_' if int(data[idx][0]) <= 2 else data[idx - 2][3],
        'is_first': data[idx][0] == 1,
        'is_last': (idx == len(data) - 1) or (int(data[idx + 1][0]) == 1),
        'is_numeric': data[idx][1].isdigit(),
    }, data[idx][3])

def create_data_frames(data):
    rows = []
    for i in range(0, len(data)):
        (features, pos_class) = get_features(data, i)
        rows.append({'features': features, 'class': pos_class})
    print 'total data', len(rows)
    return DataFrame(rows)

if __name__ == '__main__':
    load_raw_data(DATA_FILENAME)
    data_frame = create_data_frames(RAW_DATA)[:2500]

    d = DictVectorizer(sparse=False)

    pipeline = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        # ('classifier', DecisionTreeClassifier(criterion='entropy')) # ~ 6.5
        # ('classifier', RandomForestClassifier()) # ~ 6.7
        ('classifier', GaussianNB()) # ~ 5
    ])

    k_fold = KFold(n_splits=10)
    scores = []
    for train_indices, test_indices in k_fold.split(data_frame):
        # print "train", train_indices
        # print "test", test_indices
        train_features = data_frame.iloc[train_indices]['features'].values
        train_class = data_frame.iloc[train_indices]['class'].values
        # print train_features
        # print train_class
        test_features = data_frame.iloc[test_indices]['features'].values
        test_class = data_frame.iloc[test_indices]['class'].values
        pipeline.fit(train_features, train_class)
        score = pipeline.score(test_features, test_class)
        print score
        scores.append(score)

    print 'Score:', sum(scores)/len(scores)
