import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class PatchSTEmbedding(nn.Module):
    def __init__(self, emb_size, n_channels=16, kernel_size = 15, stride = 8, kernel_size2 = 15, stride2 = 8):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, emb_size, kernel_size=kernel_size2, stride=stride2),
            Rearrange("b c s -> b s c"),
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, sequence_num=1000, inter=100, n_channels=16):
        super(ChannelAttention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(
                n_channels
            ),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3),
        )
        self.key = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            # nn.LeakyReLU(),
            nn.LayerNorm(n_channels),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.attn_gradients = None
        self.attn_map = None
    
    def get_attn_map(self):
        return self.attn_map 
    
    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad 


    def forward(self, x, register_hook = False):
        temp = rearrange(x, "b c s->b s c")
        temp_query = rearrange(self.query(temp), "b s c -> b c s")
        temp_key = rearrange(self.key(temp), "b s c -> b c s")

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b c s, b m s -> b c m", channel_query, channel_key) / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        self.attn_map = channel_atten_score
        if register_hook:
            channel_atten_score.register_hook(self.save_attn_grad)

        out = torch.einsum("b c s, b c m -> b c s", x, channel_atten_score)
        """
        projections after or before multiplying with attention score are almost the same.
        """
        out = rearrange(out, "b c s -> b s c")
        out = self.projection(out)
        out = rearrange(out, "b s c -> b c s")
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.attn_gradients = None
        self.attn_map = None
        self.value = None
    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map 
    
    def get_value(self):
        return self.value

    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad 

    def forward(self, x, mask=None, register_hook = False):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        self.attn_map = att # save the attention map
        if register_hook:
            att.register_hook(self.save_attn_grad)
        self.value = values # save the values computed by the value function
        
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        ) 

class FeedForwardBlockReLU(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        ) 

class GELU(nn.Module):
    def forward(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super(TransformerEncoderBlock, self).__init__()
        #  ResidualAdd(
        #         nn.Sequential(
        #             nn.LayerNorm(emb_size),
        #             MultiHeadAttention(emb_size, num_heads, drop_p),
        #             nn.Dropout(drop_p),
        #         )
        #     ),
        #     ResidualAdd(
        #         nn.Sequential(
        #             nn.LayerNorm(emb_size),
        #             FeedForwardBlock(
        #                 emb_size, expansion=forward_expansion, drop_p=forward_drop_p
        #             ),
        #             nn.Dropout(drop_p),
        #         )
        #     ),

        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.mhattn = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout1 = nn.Dropout(drop_p)


        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 =  nn.Dropout(drop_p)
        self.mlp_outs = None

    def get_mlp_outs(self):
        return self.mlp_outs.view(self.mlp_outs.shape[0], -1)

    def forward(self, x, register_hook = False):

        # Residual Add1 - 
        res = x
        x = self.layer_norm1(x)
        x = self.mhattn(x, register_hook=register_hook)
        x = self.dropout1(x)
        x += res 
        # Residual Add2
        res = x 
        x = self.layer_norm2(x)
        x = self.ff(x)
        self.mlp_outs = x
        x = self.dropout2(x)
        x += res
        return x
    
class TransformerEncoderBlockReLU(nn.Sequential):
    def __init__(
        self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5
    ):
        super(TransformerEncoderBlockReLU, self).__init__()
        #  ResidualAdd(
        #         nn.Sequential(
        #             nn.LayerNorm(emb_size),
        #             MultiHeadAttention(emb_size, num_heads, drop_p),
        #             nn.Dropout(drop_p),
        #         )
        #     ),
        #     ResidualAdd(
        #         nn.Sequential(
        #             nn.LayerNorm(emb_size),
        #             FeedForwardBlock(
        #                 emb_size, expansion=forward_expansion, drop_p=forward_drop_p
        #             ),
        #             nn.Dropout(drop_p),
        #         )
        #     ),

        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.mhattn = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout1 = nn.Dropout(drop_p)

        
        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForwardBlockReLU(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 =  nn.Dropout(drop_p)

    def forward(self, x, register_hook = False):

        # Residual Add1 - 
        res = x
        x = self.layer_norm1(x)
        x = self.mhattn(x, register_hook=register_hook)
        x = self.dropout1(x)
        x += res 
        # Residual Add2
        res = x 
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout2(x)
        x += res
        return x

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads = 8, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads=num_heads, drop_p=dropout, forward_drop_p=dropout)
            for _ in range(depth)
        ])
        
    def forward(self, x, register_hook= False):
        for blk in self.transformer_blocks:
            x = blk(x, register_hook=register_hook)
        return x


class TransformerEncoderReLU(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads = 8, dropout=0.3):
        super(TransformerEncoderReLU, self).__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlockReLU(emb_size, num_heads=num_heads, drop_p=dropout, forward_drop_p=dropout)
            for _ in range(depth)
        ])

    def forward(self, x, register_hook= False):
        for blk in self.transformer_blocks:
            x = blk(x, register_hook=register_hook)
        return x

class STTransformer(nn.Module):
    """
    Refer to https://arxiv.org/abs/2106.11170
    Modified from https://github.com/eeyhsong/EEG-Transformer
    """

    def __init__(
        self,
        emb_size=256,
        depth=3,
        num_heads=8,
        n_classes=4,
        channel_length=2000,
        n_channels=16,
        dropout=0.3,
        kernel_size=15,
        stride=8, 
        kernel_size2 = 15,
        stride2= 8,
        **kwargs
    ):
        super().__init__()

        # convert this from residual add to straight up singular components
        # self.channel_attension = ResidualAdd(
        #     nn.Sequential(
        #         nn.LayerNorm(channel_length),
        #         ChannelAttention(n_channels=n_channels),
        #         nn.Dropout(dropout),
        #     )
        # )
        self.layer_norm = nn.LayerNorm(channel_length)
        self.channel_attention = ChannelAttention(n_channels=n_channels)
        self.dropout = nn.Dropout(dropout)

        self.patch_embedding = PatchSTEmbedding(emb_size=emb_size, n_channels = n_channels, kernel_size=kernel_size, stride=stride)
        self.transformer = TransformerEncoder(depth=depth, emb_size=emb_size, num_heads=num_heads, dropout=dropout)
        self.classification = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes)
        )
        self.avg_transformer_out = None
    
    def get_avg_transformer_out(self):
        return self.avg_transformer_out

    def forward(self, x, register_hook = False):
        # first residual add with channel attention 
        res = x 
        x = self.layer_norm(x)
        x = self.channel_attention(x, register_hook=register_hook)
        x = self.dropout(x)
        x += res

        # normal transformer sequence
        x = self.patch_embedding(x)
        # average transformer output for classification
        x = self.transformer(x, register_hook=register_hook).mean(dim=1) 
        self.avg_transformer_out = x
        x = self.classification(x)
  
        return x

class STTransformerReLU(nn.Module):
    """
    Refer to https://arxiv.org/abs/2106.11170
    Modified from https://github.com/eeyhsong/EEG-Transformer
    """

    def __init__(
        self,
        emb_size=256,
        depth=3,
        num_heads=8,
        n_classes=4,
        channel_length=2000,
        n_channels=16,
        dropout=0.3,
        kernel_size=15,
        stride=8, 
        kernel_size2 = 15,
        stride2= 8,
        **kwargs
    ):
        super().__init__()

        # convert this from residual add to straight up singular components
        # self.channel_attension = ResidualAdd(
        #     nn.Sequential(
        #         nn.LayerNorm(channel_length),
        #         ChannelAttention(n_channels=n_channels),
        #         nn.Dropout(dropout),
        #     )
        # )
        self.layer_norm = nn.LayerNorm(channel_length)
        self.channel_attention = ChannelAttention(n_channels=n_channels)
        self.dropout = nn.Dropout(dropout)

        self.patch_embedding = PatchSTEmbedding(emb_size=emb_size, n_channels = n_channels, kernel_size=kernel_size, stride=stride)
        self.transformer = TransformerEncoderReLU(depth=depth, emb_size=emb_size, num_heads=num_heads, dropout=dropout)
        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x, register_hook = False):
        # first residual add with channel attention 
        res = x 
        x = self.layer_norm(x)
        x = self.channel_attention(x, register_hook=register_hook)
        x = self.dropout(x)
        x += res

        # normal transformer sequence
        x = self.patch_embedding(x)
        # average transformer output for classification
        x = self.transformer(x, register_hook=register_hook).mean(dim=1) 
        x = self.classification(x)
  
        return x

if __name__ == "__main__":
    X = torch.randn(2, 16, 2000)
    model = STTransformer(n_classes=6)
    out = model(X)
    print(out.shape)
