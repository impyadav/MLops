"""
# -*- coding: utf-8 -*-

Created on Jan 2022
@author: Prateek Yadav

"""
import torch
import torch.nn as nn

import yaml
import numpy as np

from src.features import dataset
from src.data import textProcess
from src.modelling import model


def train():
    # config load

    with open('config/train_config.yaml') as f:
        CONFIG = yaml.safe_load(f)

    TRAIN_FILE = CONFIG['source_file']
    EPOCHS = CONFIG['EPOCHS']
    EMBED_DIM = CONFIG['Embedding_Size']
    CONTEXT_WINDOW = CONFIG['Context_Window']
    K = CONTEXT_WINDOW * 2
    MODEL_PATH = CONFIG['model_name']

    # data processing
    vocab, word_to_ix, ix_to_word, tokens = textProcess.TextFileProcess(TRAIN_FILE).create_vocab()

    # sgns data preparation
    sgns_data = dataset.CustomSGNSDataset(tokens, CONTEXT_WINDOW, K).generate_sgns_data()

    # Model initialization
    network = model.SGNS(len(vocab), EMBED_DIM)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001)

    # Model Training
    losses = []

    for epoch in range(EPOCHS):

        total_loss = 0

        for mix_data in sgns_data:
            target_word = mix_data[0][0]
            context = [item[1] for item in mix_data]

            y_label = torch.unsqueeze(torch.tensor([item[2] for item in mix_data]), 1).float()

            target_idx = torch.tensor([word_to_ix[target_word]])
            context_idxs = torch.tensor([word_to_ix[word] for word in context], dtype=torch.long)

            network.zero_grad()

            scores = network(target_idx, context_idxs)

            loss = loss_fn(torch.transpose(scores, 0, 1), y_label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            losses.append(total_loss)

        # if total_loss < np.min(losses[:-1]):
        #     print('Saving @ loss: ', total_loss)
        #     torch.save(network.state_dict(), 'models/{}'.format(MODEL_PATH))

        print('Epoch {}/{} and loss: {}'.format(epoch, EPOCHS, total_loss))

    torch.save(network.state_dict(), 'models/{}'.format(MODEL_PATH))


train()
