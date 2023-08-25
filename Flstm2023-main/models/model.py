import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer, InputAttnEncoder
from models.embed import DataEmbedding

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, \
    series_decomp_multi, FourierDecomp
import math
import numpy as np

def get_params(num_hiddens, device):
    num_inputs = num_outputs = 1

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(
                (num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))

    W_ri = normal((num_inputs, num_hiddens))
    W_rf = normal((num_inputs, num_hiddens))
    W_ro = normal((num_inputs, num_hiddens))
    W_rc = normal((num_inputs, num_hiddens))
    W_xi,W_hi,b_i = three()
    W_xf,W_hf,b_f = three()
    W_xo,W_ho,b_o = three()
    W_xc,W_hc,b_c = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)


    W_xh1,W_hh1,b_h1 = three()
    W_hq1 = normal((num_hiddens, num_outputs))
    b_q1 = torch.zeros(num_outputs, device=device)
    params = [ W_xi,W_ri,W_hi,b_i,W_xf,W_rf,W_hf,b_f,W_xo,W_ro,W_ho,b_o,W_xc,W_rc,W_hc,b_c,W_hq,b_q,W_xh1,W_hh1,b_h1,W_hq1,b_q1]
    for param in params:
        param.requires_grad_(True)
    return params

def get_params(num_hiddens, device):
    num_inputs = num_outputs = 512

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(
                (num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))

    W_ri = normal((num_inputs, num_hiddens))
    W_rf = normal((num_inputs, num_hiddens))
    W_ro = normal((num_inputs, num_hiddens))
    W_rc = normal((num_inputs, num_hiddens))
    W_xi,W_hi,b_i = three()
    W_xf,W_hf,b_f = three()
    W_xo,W_ho,b_o = three()
    W_xc,W_hc,b_c = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [ W_xi,W_ri,W_hi,b_i,W_xf,W_rf,W_hf,b_f,W_xo,W_ro,W_ho,b_o,W_xc,W_rc,W_hc,b_c,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def flstm(input,H,C,convd,W_xi,W_ri,W_hi,b_i,W_xf,W_rf,W_hf,b_f,W_xo,W_ro,W_ho,b_o,W_xc,W_rc,W_hc,b_c,W_hq,b_q):

    X = input.view(32, 512)
    I = torch.sigmoid(torch.mm(X, W_xi)  + torch.mm(H, W_hi) + b_i)#+torch.mm(R, W_ri)
    F = torch.sigmoid(torch.mm(X, W_xf)  + torch.mm(H, W_hf)+ b_f)#+torch.mm(R, W_rf)
    O = torch.sigmoid(torch.mm(X, W_xo)  + torch.mm(H, W_ho) + b_o)#+torch.mm(R, W_ro)
    C_tilda = torch.tanh(torch.mm(X, W_xc)+torch.mm(H, W_hc) + b_c) #+torch.mm(R, W_rc)
    C = F*C + I*C_tilda
    H = O*torch.tanh(C)
    CH = torch.tanh(H + C)
    CH = torch.tanh(CH + convd(CH))
    #Y = torch.mm(CH, W_hq) + b_q
    return CH, (H,C)




class LSTMModelScratch(nn.Module):
    def __init__(self, num_hiddens, device, get_params, H,C,forward_fn):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.params = get_params(num_hiddens, device)
        self.W_xi = nn.Parameter(self.params[0])
        self.W_ri = nn.Parameter(self.params[1])
        self.W_hi = nn.Parameter(self.params[2])
        self.b_i = nn.Parameter(self.params[3])
        self.W_xf = nn.Parameter(self.params[4])
        self.W_rf = nn.Parameter(self.params[5])
        self.W_hf = nn.Parameter(self.params[6])
        self.b_f = nn.Parameter(self.params[7])
        self.W_xo = nn.Parameter(self.params[8])
        self.W_ro = nn.Parameter(self.params[9])
        self.W_ho = nn.Parameter(self.params[10])
        self.b_o = nn.Parameter(self.params[11])
        self.W_xc = nn.Parameter(self.params[12])
        self.W_rc = nn.Parameter(self.params[13])
        self.W_hc = nn.Parameter(self.params[14])
        self.b_c = nn.Parameter(self.params[15])
        self.W_hq = nn.Parameter(self.params[16])
        self.b_q = nn.Parameter(self.params[17])
        self.convd = nn.Linear(512,512,bias=True)
        self.H, self.C, self.forward_fn = H, C, forward_fn
        self.linear = nn.Linear(512, 512, bias=True)
        self.linear1 = nn.Linear(512,512,bias=True)
        self.linear2 = nn.Linear(512,512,bias=True)
        self.linear3 = nn.Linear(512,512,bias=True)
        self.linear4 = nn.Linear(512,512,bias=True)
        self.linear5 = nn.Linear(512, 512, bias=True)
        self.linear6 = nn.Linear(512, 512, bias=True)
        self.linear7 = nn.Linear(512, 512, bias=True)
        self.linear8 = nn.Linear(512, 512, bias=True)
        self.linear9 = nn.Linear(512, 512, bias=True)
        self.linear10 = nn.Linear(512, 512, bias=True)



    def forward(self,i,X,H,C):
        CH, (H,C) = flstm(X,H,C,self.convd,self.W_xi,self.W_ri,self.W_hi,self.b_i,self.W_xf,self.W_rf,self.W_hf,self.b_f,self.W_xo,self.W_ro,self.W_ho,self.b_o,self.W_xc,self.W_rc,self.W_hc,self.b_c,self.W_hq,self.b_q)
        if i==-1:
            CH = self.linear(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 0:
            CH = self.linear1(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 1:
            CH = self.linear2(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 2:
            CH = self.linear3(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 3:
            CH = self.linear4(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 4:
            CH = self.linear5(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 5:
            CH = self.linear6(CH)
            CH = CH.unsqueeze(dim=0)
        elif i/24 == 6:
            CH = self.linear7(CH)
            CH= CH.unsqueeze(dim=0)
        return CH, (H,C)

class Flstm(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=2048,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Flstm, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention
        # Decomp
        kernel_size = moving_avg = 24  # 24 [24] [24,6] [24,12,6] [24,12,6,3]
        # self.decomp = series_decomp_multi(kernel_size)
        self.decomp = series_decomp(kernel_size)
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq,
                                           dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq,
                                           dropout)
        self.flstm = LSTMModelScratch(512, torch.device("cuda" if torch.cuda.is_available() else "cpu"), get_params, torch.zeros(32, 512),torch.zeros(32, 512), flstm)

        self.linear1 = nn.Linear(1,512,bias=True)
        self.linear2 = nn.Linear(512, 1, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        H = torch.zeros(32, 512).to("cuda")
        C = torch.zeros(32, 512).to("cuda")
        enc_out = self.linear1(x_enc)
        enc_out = enc_out.permute(1,0,2)
        new = 0
        for input in enc_out:
            temp,(H,C) = self.flstm(-1,input,H,C)
        for j in range(self.pred_len):
            temp,(H,C) = self.flstm(j,temp,H,C)
            if j==0:
                new = temp
            else:
                temp = temp.view(-1,32,512)
                new = torch.cat((new, temp), dim=0).view(-1,32,512)
        enc_out = new

        enc_out = enc_out.permute(1,0,2)
        dec_out = self.linear2(enc_out)
        if self.output_attention:
            return dec_out[:, :, :]
        else:
            return dec_out[:, :, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

