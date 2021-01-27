import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.distributions import Categorical

from rnn.CharDataset import CharDataset


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--chars', type=str, required=True)
    parser.add_argument('--temp', type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    characters = CharDataset.deserialize(args.chars)

    vocab_to_index = characters['vocab_to_index']
    index_to_vocab = characters['index_to_vocab']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model)
    model.eval()
    model = model.to(device)

    output = [args.start]

    state_hidden = model.zero_state(1).to(device)

    for x in range(args.count):

        current = torch.tensor([[vocab_to_index[output[-1]]]]).to(device)

        # The soft-max probabilities
        prediction, state_hidden = model(current, state_hidden)

        prediction = prediction[0]

        # Scale by the temperature
        prediction = F.softmax(prediction / args.temp, dim=1)

        dist = Categorical(prediction)
        index = dist.sample()

        # Convert the probabilities to the character
        character = index_to_vocab[index[0].item()]

        output.append(character)

    print(''.join(output))


if __name__ == '__main__':
    main()
