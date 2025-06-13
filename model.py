import torch
import torch_geometric

torch_geometric.typing.WITH_PYG_LIB = False
import numpy as np
import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops, dropout_adj
from torch.nn import Dropout, MaxPool1d, AvgPool1d
from torch_geometric.nn import Sequential, GCNConv, MixHopConv

import torch.nn as nn

import utils

HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2
EPS = 1e-8


class GTN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, pooling, residual):
        super(GTN, self).__init__()
        self.drop_rate = drop_rate
        self.pooling = pooling
        self.residual = residual
        self.independence = False
        self.convs = t.nn.ModuleList()
        mid_channels = in_channels + hidden_channels if residual else hidden_channels
        self.convs.append(
            TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                            concat=False, beta=True))
        self.ln1 = LayerNorm(in_channels=mid_channels)
        if pooling:
            self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                              dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True))
            self.ln2 = LayerNorm(in_channels=hidden_channels * heads // 2)
            self.pool = MaxPool1d(2, 2) if pooling == 'max' else AvgPool1d(2, 2) if pooling == 'avg' \
                else Linear(hidden_channels * heads, hidden_channels * heads // 2)
        else:
            self.convs.append(TransformerConv(mid_channels, hidden_channels // 2, heads=heads,
                                              dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True))
            self.ln2 = LayerNorm(in_channels=hidden_channels * heads // 2)
        self.convs_last = GCNConv(48, 1, improved=False)

    def forward(self, data):
        if isinstance(data, list):
            self.independence = True
            data = data[0]
        x = data.x
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.drop_rate, force_undirected=True,
                                             training=self.training)
        edge_attr = data.edge_attr[edge_mask]

        res = x * self.residual
        x = F.leaky_relu(self.convs[0](x, edge_index, edge_attr), negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.drop_rate, force_undirected=True,
                                             training=self.training)
        edge_attr = data.edge_attr[edge_mask]

        x = self.convs[1](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = t.squeeze(self.pool(t.unsqueeze(x, 1)), dim=1) if self.pooling else x
        x = self.ln2(x)
        if self.independence:
            return t.sigmoid(self.convs_last(x, data.edge_index))[:data.batch_size]

        return x[:data.batch_size]


class Multi_GTN(t.nn.Module):
    def __init__(self, gnn, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, num_ppi, pooling,
                 residual, learnable_weight, ):
        super(Multi_GTN, self).__init__()

        self.convs = t.nn.ModuleList()
        for _ in range(num_ppi):
            if 'GTN' in gnn:
                self.convs.append(
                    GTN(in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, pooling, residual))

        if learnable_weight:
            self.ppi_weight = t.nn.ParameterList([t.nn.Parameter(t.Tensor(1, 1)) for _ in range(num_ppi)])
            for weight in self.ppi_weight:
                t.nn.init.constant_(weight, 1)
        else:
            self.ppi_weight = t.ones(num_ppi, 1)

        self.lins = t.nn.ModuleList()
        self.lins.append(Linear(int(num_ppi * hidden_channels * heads / 2), HIDDEN_DIM,
                                weight_initializer="kaiming_uniform"))
        self.dropout = Dropout(drop_rate)
        self.lins.append(Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform"))

    def forward(self, data_tuple):
        x_list = [self.convs[i](data) for i, data in enumerate(data_tuple)]
        x = t.cat(x_list, dim=1)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        return t.sigmoid(x), x_list, self.ppi_weight


class GNNLR(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, residual):
        super(GNNLR, self).__init__()
        self.residual = residual
        mid_channels = in_channels + 3 * hidden_channels if residual else 3 * hidden_channels
        self.mix = Sequential('x, edge_index', [
            (MixHopConv(in_channels, hidden_channels), 'x, edge_index -> x'),
            nn.BatchNorm1d(3 * hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            (MixHopConv(3 * hidden_channels, hidden_channels), 'x, edge_index -> x'),

        ])

        self.linear = nn.Sequential(
            nn.Linear(mid_channels, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 48)
        )

    def forward(self, data):
        x = data.x
        res = x * self.residual
        x = self.mix(x, data.edge_index)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.linear(x)[:data.batch_size]

        return t.sigmoid(x)
        # return x




class SEFMGNN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_ppi, residual, learnable_weight, args):
        super(SEFMGNN, self).__init__()
        self.convs = t.nn.ModuleList()
        self.evidentlayer = t.nn.ModuleList()
        self.args = args
        self.l2_bias = 0
        self.epsilon = 1e-5
        self.degrees_weights = nn.Parameter(torch.ones(10))
        self.alpha = nn.Parameter(torch.tensor(0.9))

        for _ in range(num_ppi):
            self.convs.append(
                GNNLR(in_channels, hidden_channels, residual)
            )

        for _ in range(num_ppi):
            self.evidentlayer.append(
                nn.Linear(48, 2)
            )

        if learnable_weight:
            self.ppi_weight = t.nn.ParameterList([t.nn.Parameter(t.Tensor(1, 1)) for _ in range(num_ppi)])
            for weight in self.ppi_weight:
                t.nn.init.constant_(weight, 1)
        else:
            self.ppi_weight = t.ones(num_ppi, 1)

        if True:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    # nn.init.kaiming_uniform_()
                    # nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def KL(self, alpha, c):
        beta = torch.ones((1, c)).to(self.args['gpu'])
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def ce_loss(self, p, alpha, c):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1,
                      keepdim=True)

        alp = E * (1 - label) + 1
        B = self.KL(alp, c)
        return torch.mean((A + B))

    def DS_Combin_two(self, alpha1, alpha2, n_classes):
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, n_classes, 1), b[1].view(-1, 1, n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def l2(self):
        """
        模型l2计算，默认是所有参数（除了embedding之外）的平方和，
        Embedding 的 L2是 只计算当前batch用到的
        :return:

        Compute the l2 term of the model, by default it's the square sum of all parameters (except for embedding)
        The l2 norm of embedding only consider those embeddings used in the current batch
        :return:
        """
        l2 = utils.numpy_to_torch(np.array(0.0, dtype='float32'))
        l2 = l2.to(self.args['gpu'])
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if self.l2_bias == 0 and 'bias' in name:
                continue
            l2 += (p ** 2).sum()
        return l2

    def forward(self, data_tuple):
        x_list = [self.convs[i](data) for i, data in enumerate(data_tuple)]
        constraint = t.stack(x_list, dim=0).permute(1, 0, 2)
        length_loss = constraint.norm(dim=2).sum()
        x_evident = [self.evidentlayer[i](x) for i, x in enumerate(x_list)]
        x_evident = [F.softplus(x) for _, x in enumerate(x_evident)]
        alphas = [evident + 1 for evident in x_evident]
        alpha_comb = alphas[0]
        for i in range(1, len(alphas)):
            alpha_comb = self.DS_Combin_two(alpha_comb, alphas[i], 2)
        true_lab = data_tuple[0].y[:data_tuple[0].batch_size, 1]
        ce_loss = self.ce_loss(true_lab, alpha_comb, 2)
        for i in range(len(alphas)):
            ce_loss += self.ce_loss(true_lab, alphas[i], 2)
        out = torch.softmax(alpha_comb, dim=1)[:, 1].unsqueeze(1)
        # SMC部分
        gate = torch.sigmoid(self.alpha)
        entropy_reg = - (gate * torch.log(gate + EPS) + (1 - gate) * torch.log(1 - gate + EPS))
        view_probs = [torch.softmax(alpha, dim=1)[:, 1].unsqueeze(1) for alpha in alphas]
        avg_view_score = torch.stack(view_probs, dim=0).mean(dim=0)
        out = gate * out + (1 - gate) * avg_view_score
        l2_loss = self.l2()
        constraint_loss = 0.1 * l2_loss + 0.1 * length_loss + 0.1 * entropy_reg
        constraint_dict = {
            'constraint_loss': constraint_loss,
            'ce_loss': ce_loss,
            'avg_view_score': avg_view_score,
            'alpha': gate
        }
        return out, x_list, self.ppi_weight, constraint_dict


