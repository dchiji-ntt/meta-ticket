#!/usr/bin/python3

# Based on https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py

from __future__ import print_function

import os
import pickle
import requests
import tqdm

# This is from https://github.com/learnables/learn2learn/blob/master/learn2learn/data/utils.py
CHUNK_SIZE = 1 * 1024 * 1024
def download_file(source, destination, size=None):
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)

def download_pkl(google_drive_id, data_root, mode):
    filename = 'mini-imagenet-cache-' + mode
    file_path = os.path.join(data_root, filename)

    if not os.path.exists(file_path + '.pkl'):
        print('Downloading:', file_path + '.pkl')
        download_file_from_google_drive(google_drive_id, file_path + '.pkl')
    else:
        print("Data was already downloaded")

def _check_exists(root, mode):
    return os.path.exists(os.path.join(root, 'mini-imagenet-cache-' + mode + '.pkl'))

root = './'
modes = {'train', 'val', 'test'}

for mode in modes:
    if mode == 'test':
        dropbox_file_link = 'https://www.dropbox.com/s/ye9jeb5tyz0x01b/mini-imagenet-cache-test.pkl?dl=1'
    elif mode == 'train':
        dropbox_file_link = 'https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1'
    elif mode == 'val':
        dropbox_file_link = 'https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1'
    else:
        raise ('ValueError', 'Needs to be train, test or validation')

    pickle_file = os.path.join(root, 'mini-imagenet-cache-' + mode + '.pkl')
    if not _check_exists(root, mode):
        download_file(dropbox_file_link, pickle_file)
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
