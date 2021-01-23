import torch
from torch.utils import data
from math import floor


def make_one_hot(vocab_size, index):
    x = [0 for _ in range(vocab_size)]
    x[index] = 1
    return torch.tensor(x)

def cull_to_batch_size(content, batch_size):

    n = len(content)

    # Already a multiple of the batch size
    if n % batch_size == 0:
        return content

    # Divide and round down so we end up at an even number
    # then times back up so we are the nearest lower multiple of
    # batch_size
    x = floor(n / batch_size) * batch_size

    return content[: x + 1]


class CharDataset(data.Dataset):

    def __init__(self, batch_size, vocab_size, content, character_index):
        super(CharDataset, self).__init__()
        self.vocab_size = vocab_size
        self.content = cull_to_batch_size(content, batch_size)
        # -1 since we can't look forward when there is nothing left.
        self.length = len(self.content) - 1
        self.character_index = character_index

    def __getitem__(self, index):
        # Get the training one hot encoding
        train = make_one_hot(self.vocab_size, self.character_index[self.content[index]])

        # For the Cross Entropy we use the index not the one-hot encoding as the target.
        target = torch.tensor([self.character_index[self.content[index + 1]]  if index < self.length - 1else 0])

        return train.type(torch.LongTensor), target.type(torch.LongTensor)

    def __len__(self):
        return self.length
