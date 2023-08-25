import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import random

class NewAttention(nn.Module):
    def __init__(self,d_model, n_heads, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(NewAttention, self).__init__()
        print('NewAttention used !')

        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.agg = None
        self.use_wavelet = False

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        # @decor_time

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg  # size=[B, H, d, S]

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        #Q_K_sample: (B 32, H 8, L_k 96, sample_k 25)
        #torch.div(Q_K_sample.sum(-1), sample_k) : (B 32, H 8, L_K 96)
        #torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K): (B 32, H 8)
        #torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K).unsqueeze(-1).expand(B, H, L_K): (B 32, H 8)
        # 局部平均值-全局平均值 具体的思路是 用torch.div(Q_K_sample.sum(-1), sample_k)减去Q_K_sample.sum(-1).sum(-1).squeeze(-2)
        #M = Q_K_sample.sum(-1) - torch.div((torch.div(Q_K_sample.sum(-1), sample_k)).sum(-1),L_K) 然后升维
        # 原版
        #M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # JS
        #M = torch.div((Q_K_sample.max(-1)[0]-(torch.div(Q_K_sample.max(-1)[0]+torch.div(Q_K_sample.sum(-1), L_K), 2)))+(torch.div(Q_K_sample.sum(-1), L_K)-(torch.div(Q_K_sample.max(-1)[0]+torch.div(Q_K_sample.sum(-1), L_K), 2))), 2)
        #M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1),sample_k)
        #M = torch.div(Q_K_sample.sum(-1),sample_k) - Q_K_sample.min(-1)[0]
        # 最大值减最小值
        M = Q_K_sample.max(-1)[0] - Q_K_sample.min(-1)[0]
        #M = (Q_K_sample.max(-1)[0] - Q_K_sample.min(-1)[0]) - torch.div(Q_K_sample.sum(-1), L_K)
        # 效果不行
        #M = torch.div(Q_K_sample.sum(-1), sample_k) - torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K).unsqueeze(-1).expand(B, H, L_K)
        # 效果不行
        #M = Q_K_sample.max(-1)[0] - torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K).unsqueeze(-1).expand(B, H, L_K)

        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # print(queries.shape)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]


        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # size=[B, H, E, L]
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)  # size=[B, H, E, L]

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)  # [B, L, H, E], [B, H, E, L] -> [B, L, H, E]
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        #meigai
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        context.transpose(2, 1).contiguous(), attn

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))  # size = [B, L, H, E]
        else:
            return (V.contiguous(), None)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        #Q_K_sample: (B 32, H 8, L_k 96, sample_k 25)
        #torch.div(Q_K_sample.sum(-1), sample_k) : (B 32, H 8, L_K 96)
        #torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K): (B 32, H 8)
        #torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K).unsqueeze(-1).expand(B, H, L_K): (B 32, H 8)
        # 局部平均值-全局平均值 具体的思路是 用torch.div(Q_K_sample.sum(-1), sample_k)减去Q_K_sample.sum(-1).sum(-1).squeeze(-2)
        #M = Q_K_sample.sum(-1) - torch.div((torch.div(Q_K_sample.sum(-1), sample_k)).sum(-1),L_K) 然后升维
        # 原版
        #M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # JS
        #M = torch.div((Q_K_sample.max(-1)[0]-(torch.div(Q_K_sample.max(-1)[0]+torch.div(Q_K_sample.sum(-1), L_K), 2)))+(torch.div(Q_K_sample.sum(-1), L_K)-(torch.div(Q_K_sample.max(-1)[0]+torch.div(Q_K_sample.sum(-1), L_K), 2))), 2)
        #M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1),sample_k)
        #M = torch.div(Q_K_sample.sum(-1),sample_k) - Q_K_sample.min(-1)[0]
        # 最大值减最小值
        M = Q_K_sample.max(-1)[0] - Q_K_sample.min(-1)[0]
        #M = (Q_K_sample.max(-1)[0] - Q_K_sample.min(-1)[0]) - torch.div(Q_K_sample.sum(-1), L_K)
        # 效果不行
        #M = torch.div(Q_K_sample.sum(-1), sample_k) - torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K).unsqueeze(-1).expand(B, H, L_K)
        # 效果不行
        #M = Q_K_sample.max(-1)[0] - torch.div(torch.div(Q_K_sample.sum(-1), sample_k).sum(-1), L_K).unsqueeze(-1).expand(B, H, L_K)

        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class InputAttnEncoder(nn.Module):

    def __init__(self, n_feat=512, n_hidden=512, T=83):
        super(InputAttnEncoder, self).__init__()
        self.n_feat = n_feat
        self.n_hidden = n_hidden
        self.T = T

        self.lstm = nn.LSTMCell(n_feat, n_hidden)
        self.attn1 = nn.Linear(2 * n_hidden + n_feat, T + 1)
        self.attn2 = nn.Linear(T + 1, n_feat)
        self.hs_state = nn.Linear(n_feat, n_feat, bias=False)
        self.cs_state = nn.Linear(n_feat, n_feat, bias=False)
        self.x_state = nn.Linear(n_feat, n_feat, bias=False)

    # print(f'enc_out_len = {enc_out.size()}') == [32, 84, 512]

    def forward(self, X):
        # X: [n_batch, T, n_feat]=[32,84,512]
        h = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        c = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        hs, cs, atts = [], [], []
        for i in range(X.size(1)):
            xi = torch.cat([X[:, i, :], h, c], dim=1)

            #xi = torch.tanh(self.attn1(xi))
            #xi = torch.sigmoid(self.attn1(xi))
            xi = torch.relu(self.attn1(xi))
            #xi = torch.softmax(self.attn1(xi))
            xi = self.attn2(xi)
            xi = xi * X[:, i, :]
            h, c = self.lstm(xi, (h, c))
            h = self.hs_state(h)
            c = self.cs_state(c)
            #x = self.x_state(xi)
            att = torch.tanh(torch.add(h,c))
            out = torch.tanh(torch.add(att, xi))
            #print(f'h = {len(h)}')
            hs.append(h)
            cs.append(c)
            atts.append(out)

        #out = out + X
        # [n_batch, T, n_hidden]  [32,84,512]
        #hs_state = torch.stack(hs).permute(1, 0, 2)
        #print(f'hs_state == {hs_state.size()}') [32,84,512]
        #cs_state = torch.stack(cs).permute(1, 0, 2)
        atts = torch.stack(atts).permute(1, 0, 2)
        #out = torch.add(hs_state, cs_state)
        out = torch.tanh(atts)
        #hs_state_new = hs_state.reshape(32, -1)
        #cs_state_new = cs_state.reshape(32,-1)
        #input_att = self.hs_state(hs_state_new)+self.cs_state(cs_state_new)
        #print(f'hs_state_new = {hs_state_new.size()}')
        #hs_prin = hs_state_new.reshape(32, -1, 512)
        #input_att = input_att.reshape(32, -1, 512)
        #input_out = hs_state+cs_state
        #print(f'input_out == {input_out.size()}')
        #print(f'out == {out}')
        #out = torch.add(out, X)
        #out = torch.tanh(out)
        #print(f'X = {X}')
        out = X + out
        return out, torch.stack(cs).permute(1, 0, 2)

class LSTMAttention(nn.Module):

    def __init__(self, n_feat=512, n_hidden=512):
        super(LSTMAttention, self).__init__()
        self.n_feat = n_feat
        self.n_hidden = n_hidden

        self.lstm = nn.LSTMCell(n_feat, n_hidden)

        self.hs_state = nn.Linear(n_feat, n_feat, bias=False)
        self.cs_state = nn.Linear(n_feat, n_feat, bias=False)
        self.x_state = nn.Linear(n_feat, n_feat, bias=False)

    # print(f'enc_out_len = {enc_out.size()}') == [32, 84, 512]

    def forward(self, X):
        # X: [n_batch, T, n_feat]=[32,84,512]
        h = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        c = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        outs = []
        for i in range(X.size(1)):
            xi = X[:, i, :]
            h, c = self.lstm(xi, (h, c))
            h = self.hs_state(h)
            c = self.cs_state(c)
            att = torch.tanh(torch.add(h,c))
            out = torch.tanh(torch.add(att, xi))
            outs.append(out)
        outs = torch.stack(outs).permute(1, 0, 2)
        out = torch.tanh(outs)
        out = X + out
        return out

class InputAttnWithcancha(nn.Module):

    def __init__(self, n_feat=512, n_hidden=512, T=83):
        super(InputAttnWithcancha, self).__init__()
        self.n_feat = n_feat
        self.n_hidden = n_hidden
        self.T = T

        self.lstm = nn.LSTMCell(n_feat, n_hidden)
        self.attn1 = nn.Linear(2 * n_hidden + n_feat, T + 1)
        self.attn2 = nn.Linear(T + 1, n_feat)
        self.hs_state = nn.Linear(n_feat, n_feat, bias=False)
        self.cs_state = nn.Linear(n_feat, n_feat, bias=False)
        self.x_state = nn.Linear(n_feat, n_feat, bias=False)

    # print(f'enc_out_len = {enc_out.size()}') == [32, 84, 512]

    def forward(self, X):
        # X: [n_batch, T, n_feat]=[32,84,512]
        h = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        c = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        hs, cs, atts = [], [], []
        for i in range(X.size(1)):
            xi = torch.cat([X[:, i, :], h, c], dim=1)

            #xi = torch.tanh(self.attn1(xi))
            #xi = torch.sigmoid(self.attn1(xi))
            xi = torch.relu(self.attn1(xi))
            #xi = torch.softmax(self.attn1(xi))
            xi = self.attn2(xi)
            xi = xi * X[:, i, :]
            h, c = self.lstm(xi, (h, c))
            h = self.hs_state(h)
            c = self.cs_state(c)
            #x = self.x_state(xi)
            att = torch.tanh(torch.add(h,c))
            out = torch.tanh(torch.add(att, xi))
            #print(f'h = {len(h)}')
            hs.append(h)
            cs.append(c)
            atts.append(out)

        #out = out + X
        # [n_batch, T, n_hidden]  [32,84,512]
        #hs_state = torch.stack(hs).permute(1, 0, 2)
        #print(f'hs_state == {hs_state.size()}') [32,84,512]
        #cs_state = torch.stack(cs).permute(1, 0, 2)
        atts = torch.stack(atts).permute(1, 0, 2)
        #out = torch.add(hs_state, cs_state)
        out = torch.tanh(atts)
        #hs_state_new = hs_state.reshape(32, -1)
        #cs_state_new = cs_state.reshape(32,-1)
        #input_att = self.hs_state(hs_state_new)+self.cs_state(cs_state_new)
        #print(f'hs_state_new = {hs_state_new.size()}')
        #hs_prin = hs_state_new.reshape(32, -1, 512)
        #input_att = input_att.reshape(32, -1, 512)
        #input_out = hs_state+cs_state
        #print(f'input_out == {input_out.size()}')
        #print(f'out == {out}')
        #out = torch.add(out, X)
        #out = torch.tanh(out)
        #print(f'X = {X}')
        #print(out.shape)
        return out, torch.stack(cs).permute(1, 0, 2)