import os
import itertools

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f'{path} exists already')

def flatten_list(ls):
    return list(itertools.chain.from_iterable(ls))