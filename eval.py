from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')

parser.add_argument('--output', type=str,  default='output.json', help='output file')

parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()

"""

karpathy test:

tmux 0:  machine12
CUDA_VISIBLE_DEVICES=0 python eval.py --dump_images 0 --num_images 5000 --model /hdd/wangchi/Multi-Head/log/test/model.pth --infos_path /hdd/wangchi/Multi-Head/log/test/infos_GAT_enc_glu_beam5_3l_1_rl_1.pkl --language_eval 1 --beam_size 5

tmux 1:  ubuntu
CUDA_VISIBLE_DEVICES=0 python eval.py --dump_images 0 --num_images 5000 --model /home/wangchi/ImageCaption/Multi-Head/log/log_GAT_enc_glu_beam5_3l_1_rl_1/model.pth --infos_path /home/wangchi/ImageCaption/Multi-Head/log/log_GAT_enc_glu_beam5_3l_1_rl_1/infos_GAT_enc_glu_beam5_3l_1_rl_1.pkl --language_eval 1 --beam_size 5

ubuntu
CUDA_VISIBLE_DEVICES=1 python eval.py --dump_images 0 --num_images 5000 --model /home/wangchi/ImageCaption/Multi-Head/log/test/model.pth --infos_path /home/wangchi/ImageCaption/Multi-Head/log/test/infos_GAT_enc_glu_beam5_3l_1.pkl --language_eval 1 --beam_size 5

machine12
CUDA_VISIBLE_DEVICES=1 python eval.py --dump_images 0 --num_images 5000 --model /hdd/wangchi/Multi-Head/log/log_GAT_enc_glu_beam5_3l_2_rl/model.pth --infos_path /hdd/wangchi/Multi-Head/log/log_GAT_enc_glu_beam5_3l_2_rl/infos_GAT_enc_glu_beam5_3l_2_rl.pkl --language_eval 1 --beam_size 5

karpathy val:
CUDA_VISIBLE_DEVICES=2 python eval.py --dump_images 0 --num_images 5000 --model /hdd/wangchi/Multi-Head/log/log_GAT_enc_glu_beam5_3l_rl/model-best.pth --infos_path /hdd/wangchi/Multi-Head/log/log_GAT_enc_glu_beam5_3l_rl/infos_GAT_enc_glu_beam5_3l-best.pkl --language_eval 1 --beam_size 5 --split val

scp -r -P 10011 log_CIDEr1.3005/ log_CIDEr1.3012/ log_CIDEr1.3043/ wangsongchao@121.48.165.41:/hdd/wangsongchao/Multi-Head/
"""

#
# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
