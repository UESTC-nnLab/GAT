# -*- coding: utf-8 -*- 
"""
 @Author : Mason Wang 
 @E-mail : wangchiyaan@163.com
 @Time : 2020/3/6 11:34 上午 
 @File : prepro_glove.py 
 @Software: PyCharm
"""

import json

import numpy as np
from torchtext import vocab
from tqdm import tqdm

info = json.load(open('/hdd/wangchi/Multi-Head/data/cocotalk.json'))
caption_words = set(info['ix_to_word'].values())
caption_words.add('<add>')

cnt = 0
# Glove vectors
glove = vocab.GloVe(name='6B', dim=300, cache='/hdd/wangchi/caption/data')

# Get the glove vector of words for classes and attributes.
glove_vectors = []
for word in tqdm(caption_words, desc='caption_words'):
    vector = np.zeros((300))
    count = 0
    for w in word.split(' '):
        count += 1
        if w in glove.stoi:  # fetch glove vector
            glove_vector = glove.vectors[glove.stoi[w]]
            vector += glove_vector.numpy()
            cnt += 1
        else:  # use glove mean vector instead
            mean_vector = glove.vectors.mean()
            vector += mean_vector.numpy()
    glove_vectors.append(vector / count)

print('caption words in glove:', cnt)
# np.savez_compressed('./coco_glove',feat=np.vstack(glove_vectors))
np.save('./coco_glove', np.vstack(glove_vectors))
