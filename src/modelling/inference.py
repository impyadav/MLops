"""
# -*- coding: utf-8 -*-

Created on Jan 2022
@author: Prateek Yadav

"""
import yaml

import torch
import torch.nn as nn

from src.data import textProcess
from src.modelling import model


class Inference:

    def __init__(self, inputs, n_neighbors):
        self.inputs = inputs
        self.n_neighbors = n_neighbors

        with open('config/train_config.yaml') as f: self.config = yaml.safe_load(f)

        self.vocab, self.word_to_ix, self.ix_to_word, _ = textProcess.TextFileProcess(
            self.config['source_file']).create_vocab()

        self.network = model.SGNS(len(self.vocab), self.config['Embedding_Size'])
        self.network.load_state_dict(torch.load('models/{}'.format(self.config['model_name'])))

    def get_nearest_neighbors(self):
        # loading trained embedding
        all_embeds = self.network.input_embedding.weight.view(1, len(self.vocab), self.config['Embedding_Size'])

        # input_word mapping
        input_embed = self.network.input_embedding(torch.tensor([self.word_to_ix[self.inputs]]))

        # cosine similarity
        similarity_fn = nn.CosineSimilarity()

        scores = similarity_fn(input_embed, all_embeds)

        top_result = torch.topk(scores, self.n_neighbors + 1)

        pred_scores = [item for item in top_result.values.tolist()[0][1:]]
        pred_indices = [item for item in top_result.indices.tolist()[0][1:]]

        preds = [(self.ix_to_word[item], round(item1, 2)) for item, item1 in
                 zip(pred_indices[1:], pred_scores[1:])]

        return preds
