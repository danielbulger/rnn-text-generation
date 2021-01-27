import pickle
from math import floor

import torch
from torch.utils import data
from collections import Counter


class CharDataset(data.Dataset):

    @staticmethod
    def read_tokens(content):
        counts = Counter(content)
        sorted_vocab = sorted(counts, key=counts.get, reverse=True)

        index_to_vocab = {k: v for k, v in enumerate(sorted_vocab)}
        vocab_to_index = {v: k for k, v in enumerate(sorted_vocab)}

        indexed = [vocab_to_index[v] for v in content]
        return index_to_vocab, vocab_to_index, indexed

    @staticmethod
    def from_file(file_path, sequence_size):
        with open(file_path) as input_file:
            content = input_file.read()
            return CharDataset(content, sequence_size)

    @staticmethod
    def deserialize(file_path):
        with open(file_path, 'rb') as input_file:
            return pickle.load(input_file)

    def __init__(self, content, sequence_size):
        super(CharDataset, self).__init__()
        self.length = len(content) - sequence_size
        self.sequence_size = sequence_size
        self.index_to_vocab, self.vocab_to_index, self.indexed = CharDataset.read_tokens(content)

    def __getitem__(self, index):
        train = self.indexed[index: index + self.sequence_size]
        target = self.indexed[index + 1: index + 1 + self.sequence_size]
        return torch.LongTensor(train), torch.LongTensor(target)

    def __len__(self):
        return self.length

    def serialize(self, file_path):
        with open(file_path, 'wb') as output_file:
            pickle.dump({
                'count': self.get_vocab_size(),
                'index_to_vocab': self.index_to_vocab,
                'vocab_to_index': self.vocab_to_index
            }, output_file)

    def get_vocab_size(self) -> int:
        return len(self.index_to_vocab)

    def get_vocab(self, index: int) -> str:
        return self.index_to_vocab[index]

    def get_index(self, vocab: str) -> int:
        return self.vocab_to_index[vocab]
