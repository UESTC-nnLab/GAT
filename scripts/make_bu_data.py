from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
import csv
import os
import sys

import numpy as np
from torchtext import vocab
from tqdm import tqdm

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocotestbuu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)

# FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'classes', 'attrs']
# infiles = ['trainval_36_cls_attr/karpathy_test_resnet101_faster_rcnn_genome.tsv',
#            'trainval_36_cls_attr/karpathy_val_resnet101_faster_rcnn_genome.tsv',
#            'trainval_36_cls_attr/karpathy_train_resnet101_faster_rcnn_genome.tsv']
# infiles = ['trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv']
# infiles = ['trainval_resnet101_faster_rcnn_genome_36_cls_attr.tsv']
infiles = ['bottom-up_features/test2014_resnet101_faster_rcnn_genome_36_cls_attr.tsv']

if not os.path.exists(args.output_dir + '_att'):
    os.makedirs(args.output_dir + '_att')
    os.makedirs(args.output_dir + '_fc')
    os.makedirs(args.output_dir + '_box')

#
# Get classes and attributes words
data_path = './data/genome/1600-400-20/'
words = ['NO_ATTR']
# Load classes
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for obj in f.readlines():
        words.append(obj.split(',')[0].lower().strip())
# Load attributes
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        words.append(att.split(',')[0].lower().strip())
print('Classes and attributes vocab size:', len(words))

# Glove vectors
glove = vocab.GloVe(name='6B', dim=300, cache='./data')
# Get the glove vector of words for classes and attributes.
word2glove = {}
for word in tqdm(words, desc='words'):
    vector = np.zeros((300))
    count = 0
    for w in word.split(' '):
        count += 1
        if w in glove.stoi:  # fetch glove vector
            glove_vector = glove.vectors[glove.stoi[w]]
            vector += glove_vector.numpy()
        else:  # use glove mean vector instead
            mean_vector = glove.vectors.mean()
            vector += mean_vector.numpy()
    word2glove[word] = vector / count

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader, desc='images'):
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['features']:
                item[field] = np.frombuffer(base64.decodestring(item[field].encode('utf-8')),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))

            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes'].encode('utf-8')),
                dtype=np.float32).reshape((item['num_boxes'], -1))
            # bboxes: [x_min, y_min, x_max, y_max]
            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h
            area = (box_height * box_width) / (image_h * image_w)

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]
            area = area[..., np.newaxis]
            # spatial_features: [x_min, y_min, x_max, y_max, h, w, area]
            spatial_features = np.concatenate(  # 5 dims
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 area),
                axis=1)

            classes = list(
                item['classes'].split(','))
            attrs = list(
                item['attrs'].split(','))
            features_attrs_classes = []
            for i in range(item['num_boxes']):
                temp = np.concatenate([item['features'][i], word2glove[attrs[i]], word2glove[classes[i]]])
                features_attrs_classes.append(temp)

            np.savez_compressed(os.path.join(args.output_dir + '_att', str(item['image_id'])),
                                feat=np.vstack(features_attrs_classes))
            np.save(os.path.join(args.output_dir + '_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir + '_box', str(item['image_id'])), spatial_features)

print('Done!')
