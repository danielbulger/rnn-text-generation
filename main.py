from argparse import ArgumentParser

import torch
import numpy as np

from rnn.CharDataset import make_one_hot


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--chars', type=str, required=True)

    return parser.parse_args()


def read_chars(in_file):
    import pickle

    with open(in_file, 'rb') as input_file:
        return pickle.load(input_file)


def main():
    args = parse_args()

    chars = read_chars(args.chars)
    vocab_size = chars['count']
    characters = chars['characters']
    character_index = chars['character_index']

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = torch.load(args.model)
    model.eval()

    if use_cuda:
        model = model.to(device)

    output = [args.start]

    state_hidden1, state_context1 = model.zero_state(1, model.rnn1_hidden_size)
    state_hidden2, state_context2 = model.zero_state(1, model.rnn2.hidden_size)

    state_hidden1 = state_hidden1.to(device)
    state_context1 = state_context1.to(device)

    state_hidden2 = state_hidden2.to(device)
    state_context2 = state_context2.to(device)


    for x in range(args.count):
        # Stack as we need a batch dimension of 1
        current = torch.stack([make_one_hot(vocab_size, character_index[output[-1]])])

        if use_cuda:
            current = current.to(device)

        # The soft-max probabilities
        prediction, (state_hidden1, state_context1), (state_hidden2, state_context2) = model(current, (state_hidden1, state_context1), (state_hidden2, state_context2))


        # Sample from the top N most probable predictions.
        _, top_index = torch.topk(prediction, k=10)
        choices = top_index.tolist()

        # Grab one at random.
        choice = np.random.choice(choices[0])

        # Convert the probabilities to the character
        character = characters[choice]

        output.append(character)

    print(''.join(output))


if __name__ == '__main__':
    main()
