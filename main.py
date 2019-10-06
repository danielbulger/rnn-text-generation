from argparse import ArgumentParser

import torch

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

    for x in range(args.count):
        # Stack as we need a batch dimension of 1
        current = torch.stack([make_one_hot(vocab_size, character_index[output[-1]])])

        if use_cuda:
            current = current.to(device)

        # The soft-max probabilities
        prediction = model(current)

        # The argmax of the prediction
        index = prediction.cpu().argmax(1)

        # Convert the probabilities to the character
        character = characters[index]

        output.append(character)

    print(''.join(output))


if __name__ == '__main__':
    main()
