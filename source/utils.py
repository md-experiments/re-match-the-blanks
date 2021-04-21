import os
import itertools

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f'{path} exists already')

def flatten_list(ls):
    return list(itertools.chain.from_iterable(ls))

import os
import sys
import threading

class ProgressPercentage(object):
    """File upload progress bar for boto3"""
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, round(self._seen_so_far/10e5,1), round(self._size/10e5,1),
                    percentage))
            sys.stdout.flush()

def get_keys(file_key = 'aws_key_list.txt'):
    # Looks for credentials in local file
    if os.path.exists(file_key):
        with open(file_key,'r') as f:
            secrets = f.readlines()
        secrets = [s[:-1] if s.endswith('\n') else s for s in secrets]
        secrets_dict = {s.split('=')[0]:s.split('=')[1] for s in secrets}
        AWS_ACCESS_KEY = secrets_dict['AWS_ACCESS_KEY']
        AWS_SECRET_KEY = secrets_dict['AWS_SECRET_KEY']
    # Looks for credentials as environment variables (recommended)
    else:
        AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
        AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
    return AWS_ACCESS_KEY, AWS_SECRET_KEY