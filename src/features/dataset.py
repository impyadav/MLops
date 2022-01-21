"""
# -*- coding: utf-8 -*-

Created on Jan 2022
@author: Prateek Yadav

"""
import random


class CustomSGNSDataset:

    def __init__(self, list_to_tokens, context_window, k):
        self.list_of_tokens = list_to_tokens
        self.context_window = context_window
        self.k = k

    def generate_neg_samples(self, idx):
        pos_index = range(idx - self.context_window, idx + self.context_window + 1)
        updated_idxs = set(range(len(self.list_of_tokens))).difference(set(pos_index))

        return random.sample(updated_idxs, self.k)

    def generate_sgns_data(self):

        sg_data = []

        for idx in range(self.context_window, len(self.list_of_tokens) - self.context_window):

            temp = []
            pre_context = [self.list_of_tokens[idx - idx1 - 1] for idx1 in range(self.context_window)]
            post_context = [self.list_of_tokens[idx + idx1 + 1] for idx1 in range(self.context_window)]

            for context in pre_context + post_context:
                temp.append([self.list_of_tokens[idx], context, 1])

            temp += [[self.list_of_tokens[idx], self.list_of_tokens[idx1], 0] for idx1 in
                     self.generate_neg_samples(idx)]

            sg_data.append(temp)

        return sg_data
