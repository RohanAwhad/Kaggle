import gzip
import numpy as np
import os
import pickle

def load_dataset():
    stratified_dataset_filepath = os.path.join("..", "data", "stratified_dataset.gz")

    with gzip.open(stratified_dataset_filepath, "rb") as stream:
        stratified_datasets_list = pickle.load(stream)

    for ds in stratified_datasets_list:
        yield ds