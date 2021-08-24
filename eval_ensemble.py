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

"""
On coco 2014 val:
tmux 2, machine 12:
CUDA_VISIBLE_DEVICES=2 python eval_ensemble.py --dump_images 0 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.2966 CIDEr1.297 --num_images -1 --language_eval 0 --beam_size 5 --split test --output 2014val_3047_3005_3004_2966_297.json

tmux 0, machine 12:
CUDA_VISIBLE_DEVICES=0 python eval_ensemble.py --dump_images 0 --ids CIDEr1.3047 CIDEr1.3017 CIDEr1.3004 CIDEr1.3005 CIDEr1.2966 --num_images -1 --language_eval 0 --beam_size 5 --split test --output 2014val_3047_3017_3004_3005_2966.json

tmux 0, ubuntu:
CUDA_VISIBLE_DEVICES=0 python eval_ensemble.py --dump_images 0 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.297 --num_images -1 --language_eval 0 --beam_size 5 --split test --output 2014val_3047_3004_3005_297.json


On coco 2014 test:
tmux 3, machine 12:
CUDA_VISIBLE_DEVICES=3 python eval_ensemble.py --dump_images 0 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.2966 CIDEr1.297 --input_json data/cocotest.json  --input_box_dir data/cocotestbuu_box --input_fc_dir data/cocotestbuu_fc --input_att_dir data/cocotestbuu_att --input_label_h5 none --num_images -1 --language_eval 0 --beam_size 5 --split test --output 2014test_3047_3005_3004_2966_297.json

tmux 1, machine 12:                
CUDA_VISIBLE_DEVICES=1 python eval_ensemble.py --dump_images 0 --ids CIDEr1.3047 CIDEr1.3017 CIDEr1.3004 CIDEr1.3005 CIDEr1.2966 --input_json data/cocotest.json  --input_box_dir data/cocotestbuu_box --input_fc_dir data/cocotestbuu_fc --input_att_dir data/cocotestbuu_att --input_label_h5 none --num_images -1 --language_eval 0 --beam_size 5 --split test --output 2014test_3047_3017_3004_3005_2966.json


-------------------------------------------------------------------------
On Karpathy test:

tmux 0, machine 12:
CUDA_VISIBLE_DEVICES=0 python eval_ensemble.py --dump_images 0 --num_images 5000 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.297 CIDEr1.2995 --language_eval 1 --beam_size 5 --split test

tmux 1, machine 12:
CUDA_VISIBLE_DEVICES=1 python eval_ensemble.py --dump_images 0 --num_images 5000 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.3017 CIDEr1.3009 --language_eval 1 --beam_size 5 --split test

tmux 2, machine 12:
CUDA_VISIBLE_DEVICES=2 python eval_ensemble.py --dump_images 0 --num_images 5000 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.3009 CIDEr1.2966  --language_eval 1 --beam_size 5 --split test 

tmux 3, machine 12:
CUDA_VISIBLE_DEVICES=3 python eval_ensemble.py --dump_images 0 --num_images 5000 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.3019 CIDEr1.2966 --language_eval 1 --beam_size 5 --split test

tmux 4, machine 12:
CUDA_VISIBLE_DEVICES=0 python eval_ensemble.py --dump_images 0 --num_images 5000 --ids CIDEr1.3047 CIDEr1.3004 CIDEr1.3005 CIDEr1.3009 CIDEr1.2966 --language_eval 1 --beam_size 5 --split test







"""
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--ids', nargs='+', required=True, help='id of the models to ensemble')
parser.add_argument('--weights', nargs='+', required=False, default=None, help='id of the models to ensemble')

parser.add_argument('--output', type=str,  default='output.json', help='output file')
# parser.add_argument('--models', nargs='+', required=True
#                 help='path to model to evaluate')
# parser.add_argument('--infos_paths', nargs='+', required=True, help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()

model_infos = []
model_paths = []
for id in opt.ids:
    if '-' in id:
        id, app = id.split('-')
        app = '-'+app
    else:
        app = ''
    model_infos.append(utils.pickle_load(open('log_%s/infos_%s%s.pkl' %(id, id, app), 'rb')))
    model_paths.append('log_%s/model%s.pth' %(id,app))
    print('Load model from: ', 'log_%s/infos_%s%s.pkl' %(id, id, app))

# Load one infos
infos = model_infos[0]

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
for k in replace:
    setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))

vars(opt).update({k: vars(infos['opt'])[k] for k in vars(infos['opt']).keys() if k not in vars(opt)}) # copy over options from model


opt.use_box = max([getattr(infos['opt'], 'use_box', 0) for infos in model_infos])
assert max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]), 'Not support different norm_att_feat'
assert max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]), 'Not support different norm_box_feat'

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
from models.AttEnsemble import AttEnsemble

_models = []
for i in range(len(model_infos)):
    model_infos[i]['opt'].start_from = None
    model_infos[i]['opt'].vocab = vocab
    tmp = models.setup(model_infos[i]['opt'])
    tmp.load_state_dict(torch.load(model_paths[i]))
    _models.append(tmp)

if opt.weights is not None:
    opt.weights = [float(_) for _ in opt.weights]
model = AttEnsemble(_models, weights=opt.weights)
model.seq_length = opt.max_length
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

# opt.id = '+'.join([_+str(__) for _,__ in zip(opt.ids, opt.weights)])

# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    # json.dump(split_predictions, open('vis/vis.json', 'w'))
    json.dump(split_predictions, open(opt.output, 'w'))
