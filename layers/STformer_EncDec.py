import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self,configs, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.device = torch.device("cuda")
        self.use_mete = configs.use_mete
        if hasattr(configs, 'spa'):
            self.Spa = configs.spa
        else:
            self.Spa = True
        if hasattr(configs, 'pva'):
            self.pva = configs.pva
        else:
            self.pva = True
        d_ff = d_ff or 4 * d_model
        if self.pva:
            self.aq_attention = AttentionLayer( FullAttention(False, configs.factor, attention_dropout=configs.dropout,\
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads)
            self.aq_token_attention = AttentionLayer( FullAttention(False, configs.factor, attention_dropout=configs.dropout,\
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads)
            aq_token = nn.Parameter(torch.zeros(configs.n_station, configs.num_aq_token, configs.d_model),requires_grad=True)#.to(self.device)
            self.register_parameter("aq_token", aq_token)
            nn.init.kaiming_normal_(aq_token, mode='fan_in')
        if self.use_mete:
            self.mete_token_attention= AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads)
            mete_token = nn.Parameter(torch.zeros(configs.n_station, configs.num_mete_token, configs.d_model),requires_grad=True)
            self.register_parameter("mete_token", mete_token)
            nn.init.kaiming_normal_(mete_token, mode='fan_in')
        # self.mete_attention = AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads)
        
        # self.fore_token_attention= AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads)
        # self.fore_attention = AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads)

        if self.Spa:
            self.station_token_attention= AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads)
            self.station_attention= AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention), configs.d_model, configs.n_heads)

            station_token = nn.Parameter(torch.zeros(configs.num_station_token, configs.gat_node_features, configs.d_model),requires_grad=True) 
            self.register_parameter("station_token", station_token)
            nn.init.kaiming_normal_(station_token, mode='fan_in')
        # fore_token = nn.Parameter(torch.zeros(configs.n_station, configs.num_mete_token, configs.d_model),requires_grad=True)#.to(self.device)
        
        # time_token = nn.Parameter(torch.zeros(1, 1, configs.d_model),requires_grad=True)#.to(self.device)
        # self.register_parameter("time_token", time_token)
        # nn.init.kaiming_normal_(time_token, mode='fan_in')
        # self.register_parameter("fore_token", fore_token)

# mete_token = nn.Parameter(torch.zeros(B,NS, 1, self.d_model)).to(self.device)
#         fore_token = nn.Parameter(torch.zeros(B,NS, 1, self.d_model)).to(self.device)
        
        # nn.init.kaiming_normal_(fore_token, mode='fan_in')
    
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, aq_x, mete_x, fore_x,embed_timestamp,  attn_mask=None):
        

        B, N , C, D= aq_x.size()
        # n_st_token, _, _ = self.station_token.size()
        # _, n_mete_token,  _ = self.mete_token.size()

        aq_x = torch.flatten(aq_x, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
        
        # mete_token = torch.flatten(mete_token, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
        # fore_x = torch.flatten(fore_x, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
        # fore_token = torch.flatten(fore_token, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
        if self.pva:
            aq_token = self.aq_token.repeat(B,1,1)
            # channel_x, attn_aq = self.aq_token_attention(
            #     aq_token, aq_x, aq_x,
            #     attn_mask=attn_mask
            # )
            # channel_x, attn_rc = self.aq_attention(
            #     aq_x, channel_x, channel_x,
            #     attn_mask=attn_mask
            # )
            channel_x, attn_rc = self.aq_attention(
                aq_x, aq_x, aq_x,
                attn_mask=attn_mask
            )
            attn_aq=None 
        else:
            attn_aq=None 
            attn_rc=None
            channel_x = torch.zeros_like(aq_x).to(aq_x.device)
        if self.use_mete:
            mete_x = torch.flatten(mete_x, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
            # mete_token = self.mete_token.repeat(B,1,1)
            # mete_token, attn_mete = self.mete_token_attention(
            #     mete_token, mete_x, mete_x,
            #     attn_mask=attn_mask
            # )
            mete_token, attn_mete = self.mete_token_attention(
                aq_x, mete_x, mete_x,
                attn_mask=attn_mask
            )




            channel_x =  self.dropout(channel_x) + self.dropout(mete_token)  # + self.dropout(fore_token)
        else:attn_mete=None
        # mete_token, attn_rc = self.mete_attention(
        #     aq_x, mete_token, mete_token,
        #     attn_mask=attn_mask
        # )

        # fore_token = self.fore_token.repeat(B,1,1)

        # fore_token, attn_fore = self.fore_token_attention(
        #     fore_token, fore_x, fore_x,
        #     attn_mask=attn_mask
        # )

        # fore_token, attn_rc = self.fore_attention(
        #     aq_x, fore_token, fore_token,
        #     attn_mask=attn_mask
        # )


        

        aq_x = aq_x.view(B, N , C, D)
        channel_x = channel_x.view(B, N , C, D)
        # new_mete_token = new_mete_token.view(B, N , 1, D)
        # fore_token = fore_token.view(B, N , 1, D)
        

        
 
        if self.Spa:
            station_x = torch.flatten(torch.transpose(aq_x, 1, 2), start_dim=0, end_dim=1)     #station_x [BxC,N,D]
            station_token = self.station_token.unsqueeze(0).expand(B,-1,-1,-1)
            station_token = torch.flatten(torch.transpose(station_token, 1, 2), start_dim=0, end_dim=1) 
            
            # station_token, attn_station = self.station_token_attention(
            #     station_token, station_x, station_x,
            #     attn_mask=attn_mask
            # )

            # station_x, attn_rc = self.station_attention(
            #     station_x, station_token, station_token,
            #     attn_mask=attn_mask
            # )

            station_x, attn_station = self.station_token_attention(
                station_x, station_x, station_x,
                attn_mask=attn_mask
            )
            station_x = torch.transpose(station_x.view(B, C, N, D),1, 2)
            x = aq_x + self.dropout(station_x)+ self.dropout(channel_x)
        # station_token = station_token.view(B, n_st_token , C, D)
        else:
            attn_station = None
            x = aq_x + self.dropout(channel_x)

        
        # x = aq_x + self.dropout(station_x)+ self.dropout(channel_x)
        # x = x + self.dropout(station_x) #+ self.dropout(channel_x)
        # x = aq_x + self.dropout(channel_x) + self.dropout(timestamp_token) +self.dropout(station_x)
        y = x = self.norm1(x)
        y = torch.flatten(y, start_dim=1, end_dim=2)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1)).view(B, N , C, D)

        return self.norm2(x + y), [attn_aq, attn_rc, attn_mete, attn_station,  None]


class Encoder(nn.Module):
    def __init__(self, configs, enc_in ,d_model, aq_features,mete_features, ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList([
                EncoderLayer( configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ])
        conv_layers = None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = torch.nn.LayerNorm(configs.d_model)
        
        

    def forward(self, aq_x, mete_x, fore_x,embed_timestamp=None, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                aq_x,  attn = attn_layer(aq_x, mete_x, fore_x, embed_timestamp, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            aq_x = self.norm(aq_x)

        return aq_x, attns
    


class Decoder(nn.Module):
    def __init__(self, configs,   ):
        super(Decoder, self).__init__()
        Q_token = nn.Parameter(torch.zeros(configs.n_station, configs.gat_node_features, configs.d_model),requires_grad=True)#.to(self.device)
        self.register_parameter("Q_token", Q_token)
        nn.init.kaiming_normal_(Q_token, mode='fan_in')
        self.Q_attention = AttentionLayer( FullAttention(False, configs.factor, attention_dropout=configs.dropout,\
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads)
        
        self.attn_layers = nn.ModuleList([
                DecoderLayer( configs,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ])
        conv_layers = None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = torch.nn.LayerNorm(configs.d_model)

    def forward(self, aq_x,  fore_x, embed_timestamp=None, attn_mask=None):
        # x [B, L, D]
        attns = []
        B, N , C, D= aq_x.size()
        Q_token = self.Q_token.repeat(B,1,1)
        fore_x = torch.flatten(fore_x, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]

        Q_token, attn_fore = self.Q_attention(
            Q_token, fore_x, fore_x,
            attn_mask=attn_mask
        )  #channel_x [BxN,C,D]
        Q_token = Q_token.view(B, N , C, D)
        attns.append(attn_fore)
        

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                aq_x,  attn = attn_layer(aq_x, Q_token, embed_timestamp, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            aq_x = self.norm(aq_x)

        return aq_x, attns
    


class DecoderLayer(nn.Module):
    def __init__(self,configs, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.device = torch.device("cuda")
        d_ff = d_ff or 4 * d_model
        self.channel_dec_attention = AttentionLayer( FullAttention(False, configs.factor, attention_dropout=configs.dropout,\
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads)
        self.station_dec_attention= AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads)
    

        # station_token = nn.Parameter(torch.zeros(configs.num_station_token, configs.gat_node_features, configs.d_model),requires_grad=True)#.to(self.device)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, aq_x, Q_token,embed_timestamp,  attn_mask=None):
        

        B, N , C, D= aq_x.size()
        channel_x = torch.flatten(aq_x, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
        channel_Q = torch.flatten(Q_token, start_dim=0, end_dim=1)  #channel_x [BxN,C,D]
        # station_x = torch.flatten(torch.transpose(aq_x, 1, 2), start_dim=0, end_dim=1)     #station_x [BxC,N,D]
        # station_Q = torch.flatten(torch.transpose(Q_token, 1, 2), start_dim=0, end_dim=1)     #station_x [BxC,N,D]
        channel_x, attn_channel = self.channel_dec_attention(
            channel_Q, channel_x, channel_x,
            attn_mask=attn_mask
        )
        channel_x = channel_x.view(B, N , C, D)
        # attn_channel = None
        
        # station_x, attn_station = self.station_dec_attention(
        #     station_Q, station_x, station_x,
        #     attn_mask=attn_mask
        # )
        # station_x = torch.transpose(station_x.view(B, C, N, D),1, 2)
        attn_station = None

        # x = aq_x + Q_token + self.dropout(station_x) + self.dropout(channel_x)
        x = aq_x + Q_token  + self.dropout(channel_x)
        # x = aq_x + Q_token  + self.dropout(station_x)
        # x = aq_x + self.dropout(channel_x) + self.dropout(timestamp_token) +self.dropout(station_x)
        y = x = self.norm1(x)
        y = torch.flatten(y, start_dim=1, end_dim=2)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1)).view(B, N , C, D)

        return self.norm2(x + y), [attn_channel, attn_station, ]