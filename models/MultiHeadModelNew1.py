# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .AttModel import AttModel
from .TransformerModelOld import LayerNorm, attention, clones, SublayerConnection, PositionwiseFeedForward


class MyDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(MyDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask)
        return self.norm(x)


class MyDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, src_attn, feed_forward, dropout):
        super(MyDecoderLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)


class BoxEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(BoxEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, box, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, box, mask)
        return self.norm(x)


class BoxEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(BoxEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, box, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(box, box, x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


def box_attention(query_g, key_g, query_a, key_a, value_a, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''
    d_k = query_a.size(-1)
    # concat Q/K
    q = torch.cat([query_a, query_g], dim=-1)
    k = torch.cat([key_a, key_g], dim=-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) \
             / np.sqrt(d_k * 2)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value_a), p_attn


class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, dropout=0.1, dropout_glu=0.3):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linears_G = clones(nn.Linear(d_model, d_model), 2)
        self.linears_A = clones(nn.Linear(d_model, d_model), 3)
        self.linears_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # apply aoa after attention?
        self.use_glu = True
        if self.use_glu:
            self.fc_gate = nn.Linear(d_model * 2, d_model)
            self.fc_info = nn.Linear(d_model, d_model)

            # self.glu_layer = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of GLU layer
            if dropout_glu > 0:
                self.dropout_glu = nn.Dropout(p=dropout_glu)
            else:
                self.dropout_glu = lambda x: x

            # GLU doesn't need the output linear layer
            del self.linears_out
            self.linears_out = lambda x: x

    def forward(self, query_g, key_g, query_a, key_a, value_a, mask=None):
        """
        Args:
            query_g: [batch, 36, 64]
            key_g: [batch, 36, 64]
            query_a: [batch, 36, 512]
            key_a: [batch, 36, 512]
            value_a: [batch, 36, 512]
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query_a.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # perform linear operation and split into h heads
        # [batch, 8, 36, 64]
        query_g_, key_g_ = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             # transpose to get dimensions bs * h * sl * d_model
             for l, x in zip(self.linears_G, (query_g, key_g))]
        # [batch, 8, 36, 64]
        query_a_, key_a_, value_a_ = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             # transpose to get dimensions bs * h * sl * d_model
             for l, x in zip(self.linears_A, (query_a, key_a, value_a))]

        # 2) Apply attention on all the projected vectors in batch.
        # qkv :[batch, h, L, d_model/h] -->x:[b, h, L, d_model/h], attn[b, h, L, L]
        x, self.attn = box_attention(query_g_, key_g_, query_a_, key_a_, value_a_, mask=mask,
                                     dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.use_glu:
            # Apply GLU
            gate = self.fc_gate(self.dropout_glu(torch.cat([query_a, query_g], -1)))
            info = self.fc_info(self.dropout_glu(x))
            x = torch.sigmoid(gate) * info
            # gate = query_a * 8 + query_g
            # gate = query_a + query_g
            # x = self.glu_layer(self.dropout_glu(torch.cat([x, query_a], -1)))

        return self.linears_out(x)


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, norm_q=0, dropout_glu=0.3):
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

        # apply aoa after attention?
        self.use_glu = True
        if self.use_glu:
            self.fc_gate = nn.Linear(d_model, d_model)
            self.fc_info = nn.Linear(d_model, d_model)

            # self.glu_layer = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of GLU layer
            if dropout_glu > 0:
                self.dropout_glu = nn.Dropout(p=dropout_glu)
            else:
                self.dropout_glu = lambda x: x

            # GLU doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

    def forward(self, query, key, value, mask=None):
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
        # important
        query = self.norm(query)  # [batch, 1, 512]

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

        # visual sentinel
        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query_, key_, value_, mask=mask,
                                 dropout=self.dropout)
        # "Concat" using a view,   # [batch, 8, 1, 64]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # x: [batch, 1, 512]

        if self.use_glu:
            # Apply GLU
            gate = self.fc_gate(self.dropout_glu(query))
            info = self.fc_info(self.dropout_glu(x))
            x = torch.sigmoid(gate) * info
            # gate = query_a * 8 + query_g
            # gate = query_a + query_g
            # x = self.glu_layer(self.dropout_glu(torch.cat([x, query_a], -1)))

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

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, self.d_model)  # we

        # 将att_lstm的隐层映射到self—attention的维度(1024 -> 512)
        # self.h2q = nn.Linear(opt.rnn_size + opt.input_encoding_size, self.d_model)

        c = copy.deepcopy
        box_attn = BoxMultiHeadedAttention(opt.num_heads, self.d_model)
        attn = MultiHeadedDotAttention(opt.num_heads, self.d_model, project_k_v=1,
                                       scale=opt.multi_head_scale, norm_q=0)
        ff = PositionwiseFeedForward(self.d_model, 2048, 0.1)
        # Geometrical self-attention refiner
        self.vis_refiner = BoxEncoder(BoxEncoderLayer(self.d_model, box_attn, c(ff), 0.1), 3)
        # self.encoder = BoxEncoder(BoxEncoderLayer(self.d_model, box_attn, None, 0.1), 3)  # no ff

        # Self-attention module
        self.vis_mh_attention = MyDecoder(MyDecoderLayer(self.d_model, attn, c(ff), 0.1), 3)

        self.box_embed = nn.Sequential(nn.Linear(5, self.d_model),
                                       nn.ReLU(),
                                       nn.Dropout(0.3))

        # normalize the query?
        norm_q = True
        if norm_q:
            self.norm = LayerNorm(self.d_model)
        else:
            self.norm = lambda x: x

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
        # prev_h = state[0][-1]
        att_lstm_input = torch.cat([xt, fc_feats], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        # att_feats = att_feats(2048) + semantic_feat(300+300) + geometrical_feat(5)
        geo_feat = att_feats[:, :, -5:]  # [batch, 36, 5]
        geo_embd = self.box_embed(geo_feat)  # [batch, 36, 512]

        # a stack of BoxMultiHeadedAttention()
        # [batch_size, 36, 512]
        vis_refined = self.vis_refiner(p_att_feats, geo_embd, att_masks)

        # MultiHeadedDotAttention(query, key, value, mask)
        # q = self.norm(self.h2q(torch.cat([h_att, xt], -1)))
        q = self.norm(h_att)
        vis_att = self.vis_mh_attention(q, vis_refined, None)  # [batch_size, 1, 512]

        state = (torch.stack([h_att]), torch.stack([c_att]))

        return vis_att, state


class MultiHeadModelNew1(AttModel):
    def __init__(self, opt):
        super(MultiHeadModelNew1, self).__init__(opt)
        self.num_layers = 1

        del self.ctx2att
        # 2048 -> 512, 为输入self-attention做准备
        # self.ctx2att = nn.Linear(2048, opt.d_model)
        # self.ctx2att = nn.Linear(2048 + 600, opt.d_model)
        del self.att_embed
        self.att_embed = nn.Sequential(
            nn.Linear(2048 + 5, opt.d_model),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob_lm))

        self.core = MultiHeadCore(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)

        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        # p_att_feats = self.att_embed(att_feats[:, :, :2048])
        p_att_feats = self.att_embed(torch.cat([att_feats[:, :, :2048], att_feats[:, :, -5:]], -1))

        return fc_feats, att_feats, p_att_feats, att_masks

    # on machine12:
    # CUDA_VISIBLE_DEVICES=2 sh train_mh_new1.sh
    # CUDA_VISIBLE_DEVICES=2 nohup sh train_mh_new1.sh > train_trans_both_glu.log &

    # on machine:
