import gzip
import numpy as np
import os
from pickle
from sklearn.model_selection import StratifiedKFold

DATASET_DIR = os.path.join("..", "data", "df_with_embeddings.gz")

with gzip.open(DATASET_DIR, "rb") as stream:
    df = pickle.load(stream)

X = np.array(df["Review_Embeddings"].to_list()).squeeze()
y = df['Rating'].to_numpy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

stratified_datasets_list = []
for train_index, test_index in skf.split(X, y):
    y_train, y_test = y[train_index], y[test_index]
    x_train, x_test = X[train_index], X[test_index]
    stratified_datasets_list.append((x_train, x_test, y_train, y_test))

stratified_dataset_filepath = os.path.join("..", "data", "stratified_dataset.gz")
with gzip.open(stratified_dataset_filepath, "wb") as stream:
    pickle.dump(stratified_datasets_list, stream)

print("Stratified Dataset saved at", stratified_dataset_filepath)