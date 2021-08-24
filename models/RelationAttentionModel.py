from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .AttModel import AttModel
from .AttModel import pack_wrapper
from .TransformerModelOld import LayerNorm, clones


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, geo_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    # [batch, n_box, 1]
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)  # [batch, n_box, n_box]
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)  # [batch, n_box, n_box]
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    # [batch, n_box, n_box, 4]
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if geo_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)  # [batch, n_box, n_box, d_g/2]
        cos_mat = torch.cos(mul_mat)  # [batch, n_box, n_box, d_g/2]
        embedding = torch.cat((sin_mat, cos_mat), -1)  # [batch, n_box, n_box, d_g]
    else:
        embedding = position_mat
    return (embedding)


def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055

    Args:
        query: [batch, h, 1, 64]
        key: [batch, h, 36, 64]
        value: [batch, h, 36, 64]
        box_relation_embds_matrix: [batch, h, n_box, n_box]

    Returns:

    """
    N = value.size()[:2]
    dim_k = key.size(-1)
    # dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    # attention weights
    scaled_dot = torch.matmul(w_q, w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    # w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix  # [batch, h, n_box, n_box]
    w_a = scaled_dot
    # w_a = scaled_dot.view(N,N)

    # multiplying log of geometric weights by feature weights
    # w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
    if w_g is not None:
        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
    else:
        w_mn = w_a

    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn, w_v)

    return output, w_mn

#
# class BoxMultiHeadedAttention(nn.Module):
#     '''
#     Self-attention layer with relative position weights.
#     Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
#     '''
#
#     def __init__(self, h, d_model, geo_embedding=True, dropout=0.1):
#         "Take in model size and number of heads."
#         super(BoxMultiHeadedAttention, self).__init__()
#
#         assert d_model % h == 0
#         self.geo_embedding = geo_embedding
#
#         # We assume d_v always equals d_k
#         self.h = h
#         self.d_k = d_model // h
#         if self.geo_embedding:
#             self.d_g = 64
#         else:
#             self.d_g = 4
#         geo_feature_dim = self.d_g
#
#         # matrices W_q, W_k, W_v, and output projection layer
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), h)
#
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, input_query, input_key, input_value, input_box, mask=None):
#         "Implements Figure 2 of Relation Network for Object Detection"
#         if mask is not None:
#             # if len(mask.size()) == 2:
#             #     mask = mask.unsqueeze(-2)
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#
#         # single_query = 0
#         # if len(input_query.size()) == 2:
#         #     single_query = 1
#         #     input_query = input_query.unsqueeze(1)
#
#         nbatches = input_query.size(0)
#
#         # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
#         # [batch, n_box, n_box, d_g]
#         relative_geometry_embeddings = BoxRelationalEmbedding(input_box, geo_embedding=self.geo_embedding)
#         # [batch*n_box*n_box, d_g]
#         flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (input_query, input_key, input_value))]
#
#         box_size_per_head = list(relative_geometry_embeddings.shape[:3])  # [batch, n_box, n_box]
#         box_size_per_head.insert(1, 1)  # [batch, 1, n_box, n_box]
#         relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
#                                               self.WGs]
#         # [batch, h, n_box, n_box]
#         relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
#         relative_geometry_weights = F.relu(relative_geometry_weights)
#         # relative_geometry_weights = None
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
#                                          dropout=self.dropout)
#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h * self.d_k)
#
#         x = self.linears[-1](x)
#
#         # if single_query:
#         #     query = query.squeeze(1)
#         #     x = x.squeeze(1)
#         # print('x', x.shape)
#
#         return x


class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, geo_embedding=True, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.geo_embedding = geo_embedding

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.geo_embedding:
            self.d_g = 64
        else:
            self.d_g = 4
        geo_feature_dim = self.d_g

        # matrices W_q, W_k, W_v, and output projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), h)

        self.h2alpha = nn.Linear(64, 36)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            # if len(mask.size()) == 2:
            #     mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # single_query = 0
        # if len(input_query.size()) == 2:
        #     single_query = 1
        #     input_query = input_query.unsqueeze(1)

        nbatches = input_query.size(0)

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        # [batch, n_box, n_box, d_g]
        relative_geometry_embeddings = BoxRelationalEmbedding(input_box, geo_embedding=self.geo_embedding)
        # [batch*n_box*n_box, d_g]
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)

        box_size_per_head = list(relative_geometry_embeddings.shape[:3])  # [batch, n_box, n_box]
        box_size_per_head.insert(1, 1)  # [batch, 1, n_box, n_box]
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        # [batch, h, n_box, n_box]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        # print('relative_geometry_weights', relative_geometry_weights.shape)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]

        alpha = self.h2alpha(query)
        relative_geometry_weights = (alpha * relative_geometry_weights).sum(2).unsqueeze(2)
        # print('relative_geometry_weights', relative_geometry_weights.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                         dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        x = self.linears[-1](x)

        # if single_query:
        #     query = query.squeeze(1)
        #     x = x.squeeze(1)
        # print('x', x.shape)

        return x


class RelationMultiHeadCore(nn.Module):
    def __init__(self, opt):
        super(RelationMultiHeadCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v

        self.attn = BoxMultiHeadedAttention(opt.num_heads, self.d_model)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # state[0][1] is the context vector at the last step
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([xt, fc_feats, self.ctx_drop(prev_h)], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # p_att_feats = p_att_feats(1024) + geo_feat(5)
        geo_feats = p_att_feats[:, :, -5:-1]  # [batch, 36, 5]
        # geo_feats = p_att_feats[:, :, -5:]  # [batch, 36, 5]
        # BoxMultiHeadedAttention(q, k, v, box, mask)
        attn_output = self.attn(h_att, p_att_feats[:, :, :1024], p_att_feats[:, :, :1024], geo_feats, att_masks)

        lang_lstm_input = torch.cat([attn_output.squeeze(1), h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)

        return output, state


class RelationMultiHeadModel(AttModel):
    def __init__(self, opt):
        super(RelationMultiHeadModel, self).__init__(opt)
        self.num_layers = 2

        del self.ctx2att
        del self.att_embed
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size - 5),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size - 5, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        # 1024 -> 2048, 先变成2倍大，然后分成1024+1024，分别当做k,v
        # self.ctx2att = nn.Linear(opt.rnn_size + 600, 2 * opt.multi_head_scale * opt.rnn_size)

        self.core = RelationMultiHeadCore(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        # att_feats = att_feats(2048) + semantic_feat(300+300) + geometrical_feat(5)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        geo_feats = att_feats[:, :, -5:]
        # project att_feats(2048) + semantic_feat(300 + 300) to rnn_size(1024) dim
        att_feats = pack_wrapper(self.att_embed, att_feats[:, :, :2648], att_masks)  # [batch, 36, 1024]

        # Project the attention feats first to reduce memory and computation comsumptions.
        # project visual features and semantic features
        # p_att_feats = self.ctx2att(att_feats)
        p_att_feats = torch.cat([att_feats, geo_feats], dim=-1)
        return fc_feats, att_feats, p_att_feats, att_masks

    # CUDA_VISIBLE_DEVICES=0 nohup sh train_mh_new.sh > train_vis_sem_concat_glove.log &
    # CUDA_VISIBLE_DEVICES=1 sh train.sh
    # CUDA_VISIBLE_DEVICES=1 nohup sh train.sh > train_relation_1.log &
