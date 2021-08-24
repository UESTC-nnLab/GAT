# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import misc.utils as utils
from .AttModel import pack_wrapper, AttModel


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

    def __init__(self, h, d_model, dropout=0.1):
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
        dropout_glu = 0.3
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
            # # gate = query_a * 8 + query_g
            # gate = query_a + query_g
            # x = self.glu_layer(self.dropout_glu(torch.cat([x, gate], -1)))
            gate = self.fc_gate(self.dropout_glu(torch.cat([query_a, query_g], -1)))
            info = self.fc_info(self.dropout_glu(x))
            x = torch.sigmoid(gate) * info

        return self.linears_out(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, box_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.box_embed = box_embed
        self.generator = generator

    def forward(self, src, boxes, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, boxes, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, boxes, src_mask):
        return self.encoder(self.src_embed(src), self.box_embed(boxes), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
        x = self.sublayer[0](x, lambda x: self.self_attn(box, box, x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


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


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linears_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # apply aoa after attention?
        self.use_glu = True
        dropout_glu = 0.3
        if self.use_glu:
            self.glu_layer = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of GLU layer
            if dropout_glu > 0:
                self.dropout_glu = nn.Dropout(p=dropout_glu)
            else:
                self.dropout_glu = lambda x: x

            # GLU doesn't need the output linear layer
            del self.linears_out
            self.linears_out = lambda x: x

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_, key_, value_ = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query_, key_, value_, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # return self.linears[-1](x)
        if self.use_glu:
            # Apply GLU
            x = self.glu_layer(self.dropout_glu(torch.cat([x, query], -1)))

        return self.linears_out(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        bbox_attn = BoxMultiHeadedAttention(h, d_model)
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        box_embed = nn.Sequential(nn.Linear(5, d_model),
                                  nn.ReLU(),
                                  nn.Dropout(0.3))

        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(bbox_attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
            lambda x: x,  # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            box_embed,
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))

        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(2048),) if self.use_bn else ()) +
                (nn.Linear(2048, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout)

    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        att_feats, box_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        # memory = self.model.encode(att_feats, att_masks)
        memory = self.model.encode(att_feats, box_feats, att_masks)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks
        # return fc_feats[..., :1], box_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # box_feats = att_feats[:, :, -5:-1]  # [batch, 36, 4]
        box_feats = att_feats[:, :, -5:]  # [batch, 36, 5]

        att_feats = pack_wrapper(self.att_embed, att_feats[:, :, :2048], att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] = 1  # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks])
        else:
            seq_mask = None

        return att_feats, box_feats, seq, att_masks, seq_mask
        # return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, box_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        # out = self.model(att_feats, seq, att_masks, seq_mask)
        out = self.model(att_feats, box_feats, seq, att_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                                ys,
                                subsequent_mask(ys.size(1))
                                .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]

    # CUDA_VISIBLE_DEVICES=3 nohup sh train_transformer.sh > train_transformer_box.log &
    # CUDA_VISIBLE_DEVICES=2 nohup sh train_transformer.sh > train_transformer_box_glu.log &
