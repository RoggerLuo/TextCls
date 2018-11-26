import os
import pickle

def read(path):
    data = {}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    return data

def save(path,data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, True)

