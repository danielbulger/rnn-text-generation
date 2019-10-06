import torch
from torch.utils import data


def make_one_hot(vocab_size, index):
    x = [0 for _ in range(vocab_size)]
    x[index] = 1
    return torch.tensor(x)


class CharDataset(data.Dataset):

    def __init__(self, vocab_size, content, character_index):
        super(CharDataset, self).__init__()
        self.vocab_size = vocab_size
        self.content = content

        # -1 since we can't look forward when there is nothing left.
        self.length = len(content) - 1
        self.character_index = character_index

    def __getitem__(self, index):
        # Get the training one hot encoding
        train = make_one_hot(self.vocab_size, self.character_index[self.content[index]])
        target = make_one_hot(self.vocab_size, self.character_index[self.content[index + 1]])

        return train, target

    def __len__(self):
        return self.length
