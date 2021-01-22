import torch
from torch.utils import data


def make_one_hot(vocab_size, index):
    x = [0 for _ in range(vocab_size)]
    x[index] = 1
    return torch.tensor(x)


class CharDataset(data.Dataset):

    @staticmethod
    def convert_to_batch_set(content, batch_size):
        n = len(content)

        quot = n % batch_size

        # The content length is a multiple if batch size.
        if quot  == 0:
            return content

        # Otherwise find the closest lower multiple.
        n -= quot

        # +1 since we need to look ahead 1 more character
        return content[: n + 1]


    def __init__(self, batch_size, vocab_size, content, character_index):
        super(CharDataset, self).__init__()
        self.vocab_size = vocab_size
        self.content = CharDataset.convert_to_batch_set(content, batch_size)

        # -1 since we can't look forward when there is nothing left.
        self.length = len(content) - 1
        self.character_index = character_index

    def __getitem__(self, index):

        # Get the training one hot encoding
        train = make_one_hot(self.vocab_size, self.character_index[self.content[index]])

        # For the Cross Entropy we use the index not the one-hot encoding as the target.
        target = torch.tensor([self.character_index[self.content[index + 1]]])

        return train.type(torch.LongTensor), target.type(torch.LongTensor)

    def __len__(self):
        return self.length
