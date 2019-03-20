
# coding: utf-8

# **Discriminator** 
# 
# $D : R^F × R^F  → R$ 
# 
# provides the probability scores assigned to patch-summary pair $(h_i,s)$, 
# 
# where 
# 
# $h_i∈R^{F′}$ is high-level representation for each node i;
# 
# $s$ is summary vector, that containts the global information content of the whole graph.

# In[ ]:

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        #B(x1, x2) = x1*A*x2 + b
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # unsqueeze() inserts singleton dim at position given as parameter
        c_x = torch.unsqueeze(c, 1)
        #to expand this tensor to the size of the specified tensor
        c_x = c_x.expand_as(h_pl)

        #to return a tensor with all the dimensions of input of size 1 removed
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        #to concatenate the given sequences of tensors in the given dimension
        logits = torch.cat((sc_1, sc_2), 1)

        return logits

