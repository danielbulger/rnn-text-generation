import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, rnn_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.num_layers = num_layers
        self.rnn_size = rnn_size

    def zero_state(self, batch_size):
        # One for the hidden state & one for contextual state.
        return torch.zeros(self.num_layers, batch_size, self.rnn_size)

    def forward(self, x, state):
        x = self.embedding(x)

        x, hidden = self.rnn(x, state)

        x = self.decoder(x)

        return x, hidden
