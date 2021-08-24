# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .AttModel import AttModel
from .TransformerModelOld import LayerNorm, attention, clones, PositionwiseFeedForward, SublayerConnection


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, box, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, box, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, box, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, box, mask))
        return self.sublayer[1](x, self.feed_forward)


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
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

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    # attention weights
    scaled_dot = torch.matmul(w_q, w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    # w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot
    # w_a = scaled_dot.view(N,N)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)
    # print('w g', w_g.shape)
    # print('w a', w_a.shape)
    # print('w mn', w_mn.shape)
    output = torch.matmul(w_mn, w_v)
    # print('output', output.shape)
    return output, w_mn


class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding = trignometric_embedding

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        # matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), 8)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = input_query.size(0)

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = BoxRelationalEmbedding(input_box,
                                                              trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                         dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, norm_q=0):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k 
        if self.project_k_v == 0:
            # 只对query进行投影
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query_, key_, value_, mask=mask,
                                 dropout=self.dropout)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)

        return x


class MultiHeadCore(nn.Module):
    def __init__(self, opt):
        super(MultiHeadCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.d_model
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size + self.d_model, opt.rnn_size)  # h^1_t, \hat v

        # 将att_lstm的隐层映射到self—attention的维度(1024 -> 512)
        self.fc_h_att = nn.Linear(opt.rnn_size, self.d_model)

        attn = BoxMultiHeadedAttention(opt.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, 2048, 0.1)
        self.vis_refiner = Encoder(EncoderLayer(self.d_model, attn, ff, 0.1), 3)

        self.vis_mh_attention = MultiHeadedDotAttention(opt.num_heads, self.d_model, project_k_v=1,
                                                        scale=opt.multi_head_scale, norm_q=1)
        # self.sem_attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
        #                                              scale=opt.multi_head_scale, norm_q=1)
        # merge attention output to language LSTM input
        # self.merge_att = nn.Linear(self.d_model*2, self.d_model) # 不调用要注释

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.vis_refiner.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.vis_mh_attention.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        """
        Args:
            xt:
            fc_feats: [batch_size, ]
            att_feats: [batch_size, 36, 2048+600+5]
            p_att_feats: [batch_size, 36, 512]
            state:
            att_masks:
        """
        # state[0][1] is the context vector at the last step
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([xt, fc_feats, self.ctx_drop(prev_h)], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # att_feats = att_feats(2048) + semantic_feat(300+300) + geometrical_feat(5)
        geo_feat = att_feats[:, :, -5:-1]  # [batch, 36, 4]

        # a stack of BoxMultiHeadedAttention()
        # [batch_size, 36, 512]
        vis_refined = self.vis_refiner(p_att_feats, geo_feat, att_masks)

        # MultiHeadedDotAttention(query, value, key, mask)
        h_att_proj = self.fc_h_att(h_att)
        vis_att = self.vis_mh_attention(h_att_proj, vis_refined, vis_refined, None)  # [batch_size, 1, 512]

        lang_lstm_input = torch.cat([vis_att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        output = F.dropout(h_lang, self.drop_prob_lm, self.training)

        return output, state


class MultiHeadModel(AttModel):
    def __init__(self, opt):
        super(MultiHeadModel, self).__init__(opt)
        self.num_layers = 2

        del self.ctx2att
        # 1024 -> 2048, 先变成2倍大，然后分成1024+1024，分别当做k,v
        # 2048 -> 512, 为输入self-attention做准备
        self.ctx2att = nn.Linear(2048, opt.d_model)
        # self.ctx2att = nn.Linear(2048 + 600, opt.d_model)

        self.core = MultiHeadCore(opt)

        del self.att_embed

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)

        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # project visual features and semantic features
        p_att_feats = self.ctx2att(att_feats[:, :, :2048])

        return fc_feats, att_feats, p_att_feats, att_masks

    # CUDA_VISIBLE_DEVICES=0 nohup sh train_mh.sh > train_mh_rl.log &
    # CUDA_VISIBLE_DEVICES=2 sh train.sh
    # CUDA_VISIBLE_DEVICES=1 nohup sh train.sh > train_multihead_buddy_attn_concat_glove.log &
