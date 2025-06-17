import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.STformer_EncDec import Encoder, EncoderLayer,Decoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding_st,Timestamp_Embedding
import numpy as np
import argparse


class MDSTNet(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs,**kwargs):
        super(MDSTNet, self).__init__()
        configs = argparse.Namespace(**configs)
        self.device = torch.device("cuda")
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.use_fore = configs.use_fore
        self.use_mete = configs.use_mete

        self.time_c = configs.time_c
        self.d_model = configs.d_model
        self.aq_features = configs.gat_node_features
        self.mete_features = configs.mete_features
        self.enc_in = self.aq_features + self.mete_features
        # Embedding
        self.enc_embedding = DataEmbedding_st(configs.seq_len,self.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.timestamp_embedding =Timestamp_Embedding(configs.time_c,configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.class_strategy = configs.class_strategy
 
        self.encoder = Encoder( configs, self.enc_in, self.d_model,self.aq_features,self.mete_features, )   
        if self.use_fore:
            self.decoder = Decoder( configs,    )   
  
       
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # self.reconstruct_projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        self.projector = nn.Sequential(nn.Linear(configs.d_model, configs.d_model, bias=True),
                                       nn.GELU(),
                                       nn.Dropout(p=configs.dropout),
                                       nn.Linear(configs.d_model, configs.pred_len, bias=True))

        self.reconstruct_projector = nn.Sequential(nn.Linear(configs.d_model, configs.d_model, bias=True),
                                       nn.GELU(),
                                       nn.Dropout(p=configs.dropout),
                                       nn.Linear(configs.d_model, configs.seq_len, bias=True))

    def forecast(self, aq_data,mete_data, fore_data, coordinate,time_stamp=None):
        B, NS, L, C = aq_data.shape
        _,_, FL,FC = fore_data.shape       # time_stamp [B,L,TC]time_channel
        x_enc = torch.flatten(aq_data, start_dim=0, end_dim=1)
        
        fore_x = torch.flatten(fore_data, start_dim=0, end_dim=1)
        if self.use_mete:
            mete_data = torch.flatten(mete_data, start_dim=0, end_dim=1)
            x_enc = torch.concatenate([x_enc, mete_data], axis=2)
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            x_enc, means, stdev = self.norm(x_enc)
            fore_x, _,_ = self.norm(fore_data)

        x_enc = x_enc.view(B, NS, L, self.aq_features + self.mete_features )
        if self.use_fore:
            fore_x = fore_x.view(B, NS, FL, FC )
        else:
            fore_x = None
        time_stamp = time_stamp.view(B,-1,L)
        # B, NS, L, C = x_enc.shape   # B L N
        # B: batch_size;    E: d_model;   NS: num_station
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        x_enc, fore_x, Spatial_Embedding = self.enc_embedding(x_enc,fore_x, coordinate=coordinate)
        embed_timestamp= self.timestamp_embedding(time_stamp)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        
        enc_out, attns = self.encoder(x_enc[:,:,:self.aq_features,:], x_enc[:,:,-self.mete_features:,:],fore_x,embed_timestamp, attn_mask=None)
        if self.use_fore:
            dec_out, attns = self.decoder(enc_out, fore_x,embed_timestamp, attn_mask=None)

        B, N , C, D= enc_out.size()
        # B N E -> B N S -> B S N 
        if self.use_fore:
            dec_out = self.projector(dec_out.view(B*N,C,D)).permute(0, 2, 1)[:, :, :C] # filter the covariates
        else:
            dec_out = self.projector(enc_out.view(B*N,C,D)).permute(0, 2, 1)[:, :, :C] # filter the covariates
        
        # reconstructed_out =  torch.flatten(self.reconstruct_projector(x_enc), start_dim=0, end_dim=1).permute(0, 2, 1)
        reconstructed_out =  self.reconstruct_projector(enc_out.view(B*N,C,D)).permute(0, 2, 1)
        # dec_out = dec_out.view(B, N , C, D)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :self.aq_features].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :self.aq_features].unsqueeze(1).repeat(1, self.pred_len, 1))

            reconstructed_out = reconstructed_out * (stdev[:, 0, :self.aq_features].unsqueeze(1).repeat(1, self.seq_len, 1))
            reconstructed_out = reconstructed_out + (means[:, 0, :self.aq_features].unsqueeze(1).repeat(1, self.seq_len, 1))
            

        return dec_out, reconstructed_out
    def norm(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        return x_enc, means, stdev




    def forward(self, Data, mask=None):
        # x_mark_enc, x_dec, x_mark_dec = None, None, None
        
        AQStation_coordinate = Data['AQStation_coordinate'].to(self.device)

        aq_data = Data['aq_train_data'][:, :self.seq_len, :, -self.aq_features:].to(self.device) 
        aq_data = torch.transpose(aq_data, 1, 2)

        time_stamp = Data['aq_train_data'][:, :self.seq_len, 0, 1:self.aq_features:].to(self.device) 
        time_stamp = torch.transpose(time_stamp, 1, 2)


        mete_data = Data['mete_train_data'][:, :self.seq_len, :, :].to(self.device)
        mete_data = torch.transpose(mete_data, 1, 2)
        fore_data = Data['mete_train_data'][:, self.seq_len:, :, :].to(self.device)
        fore_data = torch.transpose(fore_data, 1, 2)

        

        dec_out,reconstructed_out = self.forecast(aq_data,mete_data,fore_data, AQStation_coordinate,time_stamp)



        return dec_out[:, -self.pred_len:, :],reconstructed_out  # [B, L, D]