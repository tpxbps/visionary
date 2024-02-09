import os
import json
import jsonlines
import numpy as np
import h5py
import math
import torch

class FeaturesDB(object):
    def __init__(self, img_ft_file):
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_feature(self, key):
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:].astype(np.float32)
            self._feature_store[key] = ft
        return ft

knowledge_db = FeaturesDB("../datasets/visionary/captions.hdf5")

key = '17DRP5sb8fy_10c252c90fa24ef3b698c6f54d984c5c'
knowledge_fts = []
knowledge_feature = []
for i in range(36):
    knowledge_ids = key + '_' + str(i)
    knowledge_feature.append(knowledge_db.get_feature(knowledge_ids).reshape(1, 512))

knowledge_feature = np.concatenate(knowledge_feature, axis=0)
# knowledge_fts.append(knowledge_feature.reshape(1, 36, 1, 512))
knowledge_fts.append(knowledge_feature.reshape(1, 36, 1, 512))
print(knowledge_fts)