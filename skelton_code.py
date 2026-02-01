"""
This script can be used as skelton code to read the challenge train and test
geojsons, to train a trivial model, and write data to the submission file.
"""
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import numpy as np


from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
       'Mega Projects': 5}

## Read csvs

train_df = gpd.read_file('train.geojson', index_col=0)
test_df = gpd.read_file('test.geojson', index_col=0)

## Filtering column "mail_type"
train_x = np.asarray(train_df[['geometry']].area)
train_x = train_x.reshape(-1, 1)
train_y = train_df['change_type'].apply(lambda x: change_type_map[x])

test_x = np.asarray(test_df[['geometry']].area)
test_x = test_x.reshape(-1, 1)

print(train_x.shape, train_y.shape, test_x.shape)


def cross_validation(k, model):
    n = len(train_x)
    fold_size = n // k
    idx = np.arange(n)
    acc_sum = 0
    for f in range(k):
        start = f * fold_size
        end = (f + 1) * fold_size if f < k - 1 else n

        mask_train = (idx < start) | (idx >= end)

        mini_train_x = train_x[mask_train, :]
        mini_train_y = train_y[mask_train]

        val_x = train_x[start:end, :]
        val_y = train_y[start:end]

        m = clone(model)
        m.fit(mini_train_x, mini_train_y)
        y_pred = m.predict(val_x)
        accuracy = accuracy_score(val_y, y_pred)
        acc_sum += accuracy

    print(f"Cross validation: {k} folds, mean accuracy = {acc_sum/k}")

## Train a simple OnveVsRestClassifier using featurized data
neigh = KNeighborsClassifier(n_neighbors=3)

cross_validation(15, neigh)
neigh.fit(train_x, train_y)
pred_y = neigh.predict(test_x)

print(pred_y.shape)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['change_type'])
pred_df.to_csv("knn_sample_submission.csv", index=True, index_label='Id')