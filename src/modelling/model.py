"""
# -*- coding: utf-8 -*-

Created on Jan 2022
@author: Prateek Yadav

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNS(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(SGNS, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_word, context):
        input_embedding = self.input_embedding(input_word).view(1, -1)
        context_embedding = self.context_embedding(context)

        context_embedding = torch.transpose(context_embedding, 0, 1)

        dot = torch.mm(input_embedding, context_embedding)

        scores = F.softmax(dot)

        return scores