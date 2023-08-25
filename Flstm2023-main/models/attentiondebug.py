import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt

batch_size=512
seq_len=128
head_num=8
dim_feature=64

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # batch_size, seq_len, head_num, dim_feature
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)  # 默认是没有1/sqrt(d)

        #queries=[batch_size, seq_len, head_num, dim_feature]
        #keys=[batch_size, seq_len, head_num, dim_feature]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # score的维度应该是[batch_size, head_num, seq_len, seq_len]
        if self.mask_flag:  # 默认mask_flag是True,attn_mask是None
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L,device=queries.device)  # 获取mask的上三角矩阵(去除了对角线上的值)得到的mask矩阵的维度为[batch_size, 1, seq_len, seq_len]

            scores.masked_fill_(attn_mask.mask, -np.inf)  # 将attn_mask中的mask矩阵所有值为1的位置都设置成-np.inf，且具有广播机制

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 为什么要进行dropout?
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class MyAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # batch_size, seq_len, head_num, dim_feature
        _, S, _, D = values.shape # batch_size, seq_len, head_num, dim_feature
        scale = self.scale or 1. / sqrt(E)  # 默认是没有1/sqrt(d)

        scores = torch.einsum("bshk,bshv->bhdv",keys,values)  # score的维度应该是[batch_size, head_num, dim_feature, dim_feature]
        if self.mask_flag:  # 默认mask_flag是True,attn_mask是None
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L,device=queries.device)  # 获取mask的上三角矩阵(去除了对角线上的值)得到的mask矩阵的维度为[batch_size, 1, seq_len, seq_len]

            scores.masked_fill_(attn_mask.mask, -np.inf)  # 将attn_mask中的mask矩阵所有值为1的位置都设置成-np.inf，且具有广播机制

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 为什么要进行dropout?
        V = torch.einsum("bshd,bhdf->bhdf", queries,A)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


#attn = MyAttention(mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False)

queries=torch.randn(batch_size, seq_len, head_num, dim_feature)
keys=torch.randn(batch_size, seq_len, head_num, dim_feature)
values=torch.randn(batch_size, seq_len, head_num, dim_feature)
#attn_mask=[]

A1 = torch.einsum("blhe,bshe->bhls", queries, keys)
print(A1.shape)

V1 = torch.einsum("bhls,bshd->blhd", A1, values)
print(V1.shape)

A2 = torch.einsum("bshk,bshv->bhkv",keys,values)
print(A2.shape)

V2 = torch.einsum("bshk,bhkv->bshv",queries,A2)
print(V2.shape)
#x,a = attn(queries, keys, values,attn_mask)
#print(x.shape)

#a = torch.tensor([[ 1,  3, 5],[ 2,  4,  6]]) #[2.3]
#print(a.shape)
#b = torch.tensor([[ 1,  2],[ 3,  4],[ 5,  6]])    #[3.2]
#print(b.shape)
#c = torch.tensor([[ 1,  2, 3],[ 4, 5,  6]])  #[2.3]
#print(c.shape)

#e = torch.einsum("ik,jk->ij",a,c)
#print(e,e.shape)
