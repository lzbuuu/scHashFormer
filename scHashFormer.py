import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.nn.conv import GCNConv

from utils import dis_fun


class model(nn.Module):
    def __init__(self, cfg, input_dim, pe_dim=0):
        super(model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = cfg.hidden_dim
        self.prob_feature = cfg.prob_feature
        self.prob_edge = cfg.prob_edge
        self.alpha = cfg.alpha
        self.hops = cfg.hops
        dropout_rate = cfg.dropout_rate
        attention_dropout_rate = cfg.attention_dropout_rate

        # MLP layer for input features
        self.input_layer = nn.Linear(input_dim+pe_dim, self.hidden_dim)  # input_dim+pe_dim
        self.hash_layer = LSH(input_dim, cfg.n_buckets, cfg.n_hashes)
        self.reconstruction_layer = nn.Linear(cfg.n_buckets, input_dim)
        self.transformer = TransformerModel(cfg.hidden_dim, cfg.hidden_dim, cfg.hops, cfg.n_layers,
                                            cfg.n_heads, dropout_rate, attention_dropout_rate)
        self.predictor = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        self.centers = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden_dim))

    def forward(self, x, aug_x):
        x = self.input_layer(x)
        aug_x = self.input_layer(aug_x)
        z = self.transformer(x)
        aug_z = self.transformer(aug_x)
        s = self.predictor(z)
        s_aug = self.predictor(aug_z)

        logit = torch.cat((s.sum(1), s_aug.sum(1)), 0)
        return logit

    def forward_dict(self, x, aug_x, index_mapping):
        z = torch.empty(x.size(0), self.hidden_dim, device=x.device)
        aug_z = torch.empty(aug_x.size(0), self.hidden_dim, device=x.device)
        x = self.input_layer(x)
        aug_x = self.input_layer(aug_x)
        for i in index_mapping.keys():
            z[index_mapping[i]] = self.transformer(x[index_mapping[i]])
            aug_z[index_mapping[i]] = self.transformer(aug_x[index_mapping[i]])
        s = self.predictor(z)
        s_aug = self.predictor(aug_z)
        logit = torch.cat((s.sum(1), s_aug.sum(1)), 0)
        return logit

    def hash_loss(self, x, h, lambda_reg=0.1):
        # Calculate balance loss
        mean_per_bit = torch.mean(h, dim=0)  # 沿batch维度求均值
        ba_loss = torch.sum(torch.square(mean_per_bit))
        # Calculate reconstruction loss
        x_hat = self.reconstruction_layer(h)
        re_loss = F.mse_loss(x, x_hat)
        return re_loss + lambda_reg * ba_loss  # Combine balance loss and reconstruction loss

    def hash(self, x):
        h = self.hash_layer(x)  # Hashing the input features

        return h

    def embedding(self, x):
        x = self.input_layer(x)
        z = self.transformer(x)
        z = z + x[:, 0, :]
        z = F.normalize(z)
        return z

    def embedding_dict(self, x, index_mapping):
        z = torch.empty(x.size(0), self.hidden_dim, device=x.device)
        x = self.input_layer(x)
        for i in index_mapping.keys():
            z[index_mapping[i]] = self.transformer(x[index_mapping[i]])
        z = z + x
        z = F.normalize(z)
        return z


class LSH(nn.Module):
    def __init__(self, input_dim, num_buckets, num_hashes):
        super(LSH, self).__init__()
        buckets_size = num_buckets // num_hashes  # Size of each bucket
        layers = [nn.Linear(input_dim, buckets_size, bias=True) for _ in range(num_hashes)]  # num_hashes encoders
        self.layers = nn.ModuleList(layers)  # num_hashes layers

    def forward(self, x):
        hashes = []
        for layer in self.layers:
            h = F.sigmoid(layer(x))  # Apply tanh activation
            # h = torch.round(x)  # Round to nearest integer
            hashes.append(h)
        hashes = torch.cat(hashes, dim=1)  # Concatenate along the feature dimension
        return hashes


class TransformerModel(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=512, hops=3, n_layers=1, num_heads=8, dropout_rate=0.0, attention_dropout_rate=0.1):
        super().__init__()
        self.seq_len = hops + 1
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.is_projection = True
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.output_dim)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        # transformer encoder
        if len(batched_data.shape) == 2:
            for enc_layer in self.layers:
                z1 = enc_layer(batched_data)

            output = self.final_ln(z1)
            emb = output
        else:
            self.seq_len = batched_data.shape[1]
            for enc_layer in self.layers:
                z1 = enc_layer(batched_data)

            output = self.final_ln(z1)

            target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
            split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)

            node_tensor = split_tensor[0]
            neighbor_tensor = split_tensor[1]

            layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=-1))
            layer_atten = F.softmax(layer_atten, dim=1)
            neighbor_tensor = neighbor_tensor * layer_atten
            neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
            emb = (node_tensor + neighbor_tensor).squeeze()

        output = self.Linear1(torch.relu(self.out_proj(emb)))
        return emb

    def projection(self, x):
        z1 = self.att_embeddings_nope(x)
        self.is_projection = False
        return z1


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        # Local attention with neighbors
        norm_x = self.self_attention_norm(x)
        y = self.self_attention(norm_x, norm_x, norm_x, attn_bias)
        y = self.self_attention_dropout(y)
        z1 = norm_x + y
        y = self.ffn_norm(z1)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        z1 = z1 + y
        return z1


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        if len(orig_q_size) == 2:
            q_size = batch_size = 1
            k_size = 1
            v_size = 1
        else:
            q_size = batch_size = q.size(0)
            k_size = k.size(0)
            v_size = v.size(0)

        d_k = self.att_size
        d_v = self.att_size

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(q_size, -1, self.num_heads, d_k)  # (q_size, q_len, n_heads, d_k)
        k = self.linear_k(k).view(k_size, -1, self.num_heads, d_k)  # (k_size, k_len, n_heads, d_k)
        v = self.linear_v(v).view(v_size, -1, self.num_heads, d_v)  # (v_size, v_len, n_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        if len(orig_q_size) == 2:
            x = x.squeeze()
        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * cl_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * cl_loss(z_aug, z, adj_aug, tau, hidden_norm)
    return loss


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def cl_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)

def contrastive_loss(z, z_aug, tau, alpha1, alpha2, hidden_norm=True):
    def loss(x1, x2, tau=0.8, hidden_norm=True):
        B = x1.size(0)
        if hidden_norm:
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
        f = lambda x: torch.exp(x / tau)
        sim11 = f(x1.matmul(x1.T))
        sim12 = f(x1.matmul(x2.T))
        numerator = sim12.diag()
        denominator = sim11.sum(1) + sim12.sum(1) - sim11.diag()
        l = -(torch.log(numerator / denominator)) / B
        return l.mean()

    return alpha1*loss(z, z_aug, tau, hidden_norm) + alpha2*loss(z_aug, z, tau, hidden_norm)