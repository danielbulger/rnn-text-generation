import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, device, vocab_size, embedding_size, rnn1_size, rnn2_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn1 = nn.LSTM(embedding_size, rnn1_size, 1, batch_first=True)
        self.rnn2 = nn.LSTM(rnn1_size, rnn2_size, 1, batch_first=True)
        self.rnn1_hidden_size = rnn1_size
        self.rnn2_hidden_size = rnn2_size
        self.dense = nn.Linear(embedding_size + rnn1_size + rnn2_size, vocab_size)
        self.device = device

    def _init_hidden(self, batch_size, size):
        return torch.zeros(1, batch_size, size).to(self.device)

    def forward(self, x):
        batch_size = x.size(0)

        embedding = self.embedding(x)
        hidden1 = self._init_hidden(batch_size, self.rnn1_hidden_size)
        context1 = self._init_hidden(batch_size, self.rnn1_hidden_size)

        hidden2 = self._init_hidden(batch_size, self.rnn2_hidden_size)
        context2 = self._init_hidden(batch_size, self.rnn2_hidden_size)

        rnn1, (hidden1, context1) = self.rnn1(embedding, (hidden1, context1))
        rnn2, (hidden2, context2) = self.rnn2(rnn1, (hidden2, context2))

        embedding = embedding.transpose(0, 1)[-1]
        rnn1 = rnn1.transpose(0, 1)[-1]
        rnn2 = rnn2.transpose(0, 1)[-1]

        x = torch.cat([embedding, rnn1, rnn2], dim=1)
        x = self.dense(x)

        return F.softmax(x, dim=1)
